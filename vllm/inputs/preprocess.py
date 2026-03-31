# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import assert_never

from vllm.config import ModelConfig, ObservabilityConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalEncDecInputs,
    MultiModalInputs,
    MultiModalUUIDDict,
)
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.tokenizers import TokenizerLike
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.metrics.stats import MultiModalCacheStats

from .data import (
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ExplicitEncoderDecoderPrompt,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    embeds_inputs,
    token_inputs,
)
from .parse import is_explicit_encoder_decoder_prompt, parse_singleton_prompt

logger = init_logger(__name__)


class InputPreprocessor:
    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: TokenizerLike | None,
        observability_config: ObservabilityConfig | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        mm_processor_cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.observability_config = observability_config
        self.mm_registry = mm_registry
        self.mm_processor_cache = mm_processor_cache

        self.mm_cache_stats = MultiModalCacheStats() if mm_processor_cache else None

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "You cannot pass text prompts when `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    def get_decoder_start_token_id(self) -> int | None:
        """
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        """

        if not self.model_config.is_encoder_decoder:
            logger.warning_once(
                "Using None for decoder start token id because "
                "this is not an encoder/decoder model."
            )
            return None

        if self.model_config is None or self.model_config.hf_config is None:
            logger.warning_once(
                "Using None for decoder start token id because "
                "model config is not available."
            )
            return None

        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )
        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token "
                "id because decoder start token id is not "
                "available."
            )
            dec_start_token_id = self.get_bos_token_id()

        return dec_start_token_id

    def _get_default_enc_dec_decoder_prompt(self) -> list[int]:
        """
        Specifically for encoder/decoder models:
        generate a default decoder prompt for when
        the user specifies only the encoder prompt.

        Encoder/decoder models utilize the decoder
        prompt in different ways; as new models are
        added, it is intended that this function
        will be extended to produce differing
        default decoder prompts, depending on the
        model variety.

        Absent a special case, the default behavior
        of this method is to mirror the behavior of
        the HuggingFace (HF) GenerationMixin for a None
        decoder prompt, which is to employ a logit processor
        setting to force the first decoded token to be <BOS>.
        Here, this behavior is approximated by having the
        "default" decoder prompt be <BOS>.

        However, it is possible that in the future
        other models may have different or more
        complex logic for the default decoder prompt.
        This motivates having a special helper method
        for default decoder prompts.

        Returns:

        * prompt_token_ids
        """

        bos_token_id = self.get_bos_token_id()
        assert bos_token_id is not None
        return [bos_token_id]

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: list[int] | None,
    ) -> list[int]:
        """
        Prepares `decoder_input_ids` for generation with encoder-decoder models.

        Based on:
        https://github.com/huggingface/transformers/blob/4037a2b5b1278736e566aec12e169100275545ea/src/transformers/generation/utils.py
        specifically,
        `GenerationMixin._prepare_decoder_input_ids_for_generation()`.

        Arguments:

        * decoder_input_ids: input token ids to preprocess

        Returns:

        * Processed token list
        """

        decoder_start_token_id = self.get_decoder_start_token_id()
        assert decoder_start_token_id is not None

        if decoder_input_ids is None:
            # no decoder prompt input ->
            # use decoder_start_token_id as decoder_input_ids
            decoder_input_ids = self._get_default_enc_dec_decoder_prompt()

        if (
            len(decoder_input_ids) == 0
            or decoder_input_ids[0] != decoder_start_token_id
        ):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _get_tokenization_kw(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = dict[str, Any]()

        if self.model_config.is_encoder_decoder:
            # For Whisper, special tokens should be provided by the user based
            # on the task and language of their request. Also needed to avoid
            # appending an EOS token to the prompt which disrupts generation.
            kwargs["add_special_tokens"] = False

        if overrides:
            kwargs.update(overrides)

        return kwargs

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        logger.info(f"[SxlAdd] 开始编码文本提示")
        tokenizer = self.get_tokenizer()
        tokenization_kwargs = self._get_tokenization_kw(tokenization_kwargs)
        logger.info(f"[SxlAdd] 获取tokenizer完成，使用参数: {tokenization_kwargs}")

        encoder_config = self.model_config.encoder_config

        if encoder_config and encoder_config.get("do_lower_case", False):
            logger.info(f"[SxlAdd] 执行小写转换")
            prompt = prompt.lower()

        token_ids = tokenizer.encode(prompt, **tokenization_kwargs)
        logger.info(f"[SxlAdd] 文本编码完成，token数量: {len(token_ids)}")
        return token_ids

    def _get_mm_processor(self) -> BaseMultiModalProcessor:
        if not hasattr(self, "_mm_processor"):
            self._mm_processor = self.mm_registry.create_processor(
                self.model_config,
                self.observability_config,
                tokenizer=self.tokenizer,
                cache=self.mm_processor_cache,
            )

        return self._mm_processor

    def _process_multimodal(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object] | None,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        logger.info(f"[SxlAdd] 开始处理多模态提示")
        mm_processor = self._get_mm_processor()
        logger.info(f"[SxlAdd] 获取多模态处理器完成")

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        logger.info(f"[SxlAdd] 应用多模态处理器")
        logger.info(f"[SxlAdd] 多模态数据类型: {type(mm_data).__name__}")
        logger.info(f"[SxlAdd] 多模态数据内容: {mm_data}")
        mm_input = mm_processor.apply(
            prompt,
            mm_data,
            hf_processor_mm_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        logger.info(f"[SxlAdd] 多模态处理器应用完成")
        logger.info(f"[SxlAdd] 多模态输出类型: {type(mm_input).__name__}")
        logger.info(f"[SxlAdd] 多模态输出键: {list(mm_input.keys())}")
        if "prompt_token_ids" in mm_input:
            logger.info(f"[SxlAdd] 处理后的token数量: {len(mm_input['prompt_token_ids'])}")
            logger.info(f"[SxlAdd] 前10个token: {mm_input['prompt_token_ids'][:10]}")
        if "mm_kwargs" in mm_input:
            logger.info(f"[SxlAdd] 多模态参数类型: {type(mm_input['mm_kwargs']).__name__}")
            logger.info(f"[SxlAdd] 多模态参数键: {list(mm_input['mm_kwargs'].keys())}")
        if "mm_hashes" in mm_input:
            logger.info(f"[SxlAdd] 多模态哈希类型: {type(mm_input['mm_hashes']).__name__}")
        if "mm_placeholders" in mm_input:
            logger.info(f"[SxlAdd] 多模态占位符类型: {type(mm_input['mm_placeholders']).__name__}")
        mm_hashes = mm_input["mm_hashes"]

        # Validate that all mm items have a string as their hash
        contains_only_strings = all(
            isinstance(leaf, str) for leaf in json_iter_leaves(mm_hashes)
        )
        if not contains_only_strings:
            raise ValueError(
                f"mm_hashes must contain only strings, got: {mm_hashes}. "
                "This is likely due to an incorrect custom implementation of "
                "MultiModalProcessor.apply method."
            )

        logger.info(f"[SxlAdd] 多模态提示处理完成")
        return mm_input

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInputs:
        logger.info(f"[SxlAdd] 开始处理嵌入提示")
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = parsed_content["prompt_embeds"]
        logger.info(f"[SxlAdd] 嵌入形状: {prompt_embeds.shape}")

        # prompt_embeds must be (seq_len, hidden_size), but if the user
        # passes in a batch of size 1, i.e. (1, seq_len, hidden_size),
        # we can unambiguously process the intent by squeezing the batch
        # dimension.
        if prompt_embeds.ndim == 3:
            logger.info(f"[SxlAdd] 嵌入维度为3，进行压缩")
            prompt_embeds = prompt_embeds.squeeze(dim=0)
            logger.info(f"[SxlAdd] 压缩后形状: {prompt_embeds.shape}")

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        # Tensors must be on CPU for serialization between processes
        # in the MsgpackEncoder. Casting to CPU here ensures that there is no
        # hidden device transfer in the critical path of generation.
        logger.info(f"[SxlAdd] 将嵌入移至CPU")
        prompt_embeds = prompt_embeds.cpu()

        result = embeds_inputs(
            prompt_embeds=prompt_embeds, cache_salt=parsed_content.get("cache_salt")
        )
        logger.info(f"[SxlAdd] 嵌入提示处理完成")
        return result

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        if (
            not tokenization_kwargs
            or "truncation" not in tokenization_kwargs
            or self.tokenizer is None
        ):
            return inputs

        max_length = tokenization_kwargs["max_length"]

        if self.tokenizer.truncation_side == "left":
            return inputs[-max_length:]
        else:
            return inputs[:max_length]

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokenInputs | MultiModalInputs:
        prompt_token_ids = parsed_content["prompt_token_ids"]
        logger.info(f"[SxlAdd] 开始处理token提示，原始token数量: {len(prompt_token_ids)}")
        prompt_token_ids = self._truncate_inputs(
            prompt_token_ids, tokenization_kwargs
        )
        logger.info(f"[SxlAdd] token截断完成，数量: {len(prompt_token_ids)}")

        inputs: TokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            logger.info(f"[SxlAdd] token提示包含多模态数据")
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        else:
            logger.info(f"[SxlAdd] 处理纯token提示")
            inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt
            logger.info(f"[SxlAdd] 添加缓存盐值")

        logger.info(f"[SxlAdd] token提示处理完成，返回类型: {type(inputs).__name__}")
        return inputs

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokenInputs | MultiModalInputs:
        prompt_text = parsed_content["prompt"]
        logger.info(f"[SxlAdd] 开始处理文本提示，长度: {len(prompt_text)}, 内容: {prompt_text}")

        inputs: TokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            logger.info(f"[SxlAdd] 文本提示包含多模态数据")
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        else:
            logger.info(f"[SxlAdd] 处理纯文本提示")
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            logger.info(f"[SxlAdd] 文本编码完成，token数量: {len(prompt_token_ids)}")
            inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt
            logger.info(f"[SxlAdd] 添加缓存盐值")

        logger.info(f"[SxlAdd] 文本提示处理完成，返回类型: {type(inputs).__name__}")
        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        logger.info(f"[SxlAdd] 开始转换提示为LLM输入")
        parsed = parse_singleton_prompt(prompt)
        logger.info(f"[SxlAdd] 提示解析完成，类型: {parsed['type']}")

        if parsed["type"] == "embeds":
            logger.info(f"[SxlAdd] 处理嵌入提示")
            return self._process_embeds(parsed["content"])
        if parsed["type"] == "tokens":
            logger.info(f"[SxlAdd] 处理token提示")
            return self._process_tokens(
                parsed["content"],
                mm_uuids=mm_uuids,
            )
        if parsed["type"] == "text":
            logger.info(f"[SxlAdd] 处理文本提示")
            return self._process_text(
                parsed["content"],
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        if parsed["type"] == "str":
            logger.info(f"[SxlAdd] 处理字符串提示")
            return self._process_text(
                TextPrompt(prompt=parsed["content"]),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(parsed)

    def _build_enc_dec_llm_inputs(
        self,
        encoder_inputs: SingletonInputs,
        decoder_inputs: SingletonInputs | None,
    ) -> EncoderDecoderInputs:
        if (
            encoder_inputs["type"] == "embeds"
            or decoder_inputs
            and decoder_inputs["type"] == "embeds"
        ):
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        # Needed for mypy
        encoder_inputs = cast(TokenInputs | MultiModalInputs, encoder_inputs)
        decoder_inputs = cast(TokenInputs | MultiModalInputs | None, decoder_inputs)

        if decoder_inputs is None:
            if self.model_config.hf_config.model_type == "whisper":
                # For Whisper models, the text prompt should go to the decoder.
                # If no explicit encoder/decoder inputs, then copy the prompt
                # from the encoder to the decoder. The encoder tokens are later
                # overridden by the audio features.
                dec_token_ids = encoder_inputs["prompt_token_ids"].copy()
            else:
                dec_token_ids = self._prepare_decoder_input_ids_for_generation(None)
            decoder_inputs = token_inputs(dec_token_ids)
        else:
            if "multi_modal_data" in decoder_inputs:
                raise ValueError(
                    "Multi-modal decoder inputs of encoder-"
                    "decoder models are not supported yet"
                )

            dec_token_ids = self._prepare_decoder_input_ids_for_generation(
                decoder_inputs["prompt_token_ids"]
            )
            decoder_inputs["prompt_token_ids"] = dec_token_ids

        return EncoderDecoderInputs(
            encoder=encoder_inputs,
            decoder=decoder_inputs,
        )

    def _split_enc_dec_mm_inputs(
        self,
        inputs: SingletonInputs | MultiModalEncDecInputs,
        decoder_inputs_to_override: SingletonInputs | None = None,
    ) -> tuple[SingletonInputs, SingletonInputs]:
        """
        For encoder/decoder models only:
        Separate Encoder/Decoder inputs from a MultiModalEncDecInputs
        """
        if (
            inputs["type"] == "embeds"
            or decoder_inputs_to_override
            and decoder_inputs_to_override["type"] == "embeds"
        ):
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        # Needed for mypy
        inputs = cast(
            TokenInputs | MultiModalInputs | MultiModalEncDecInputs,
            inputs,
        )
        decoder_inputs_to_override = cast(
            TokenInputs | MultiModalInputs | None,
            decoder_inputs_to_override,
        )

        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs

        if inputs["type"] == "multimodal":  # Multimodal data inputs
            if "encoder_prompt_token_ids" not in inputs:
                raise RuntimeError(
                    "You should register an encoder-decoder "
                    "multi-modal processor for encoder-decoder "
                    "models."
                )
            inputs = cast(MultiModalEncDecInputs, inputs)

            encoder_inputs = token_inputs(inputs["encoder_prompt_token_ids"])

            decoder_prompt_inputs = decoder_inputs_to_override or inputs
            decoder_inputs = MultiModalInputs(
                type="multimodal",
                prompt_token_ids=decoder_prompt_inputs["prompt_token_ids"],
                mm_kwargs=inputs["mm_kwargs"],
                mm_hashes=inputs["mm_hashes"],
                mm_placeholders=inputs["mm_placeholders"],
            )
            if cache_salt := inputs.get("cache_salt"):
                decoder_inputs["cache_salt"] = cache_salt

        elif inputs["type"] == "token":  # Text-only inputs
            encoder_inputs = token_inputs(prompt_token_ids=[])
            decoder_inputs = decoder_inputs_to_override or inputs
        else:
            assert_never(inputs)  # type: ignore[arg-type]

        return encoder_inputs, decoder_inputs

    def _process_encoder_decoder_prompt(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderDecoderInputs:
        """
        For encoder/decoder models only:
        Process an input prompt into an
        [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
        instance.

        There are two types of input prompts:
        singleton prompts which carry only the
        encoder prompt, and explicit encoder/decoder
        prompts which carry both the encoder and the
        decoder prompts as member variables.

        This function handles the following scenarios:
        * Singleton encoder prompt: extract encoder prompt
          token ids & infer default decoder prompt token ids
        * Explicit encoder/decoder prompt: extract encoder
          and decoder prompt token ids

        Note that for Explicit encoder/decoder prompts,
        each sub-prompt (encoder or decoder prompt) can
        have any possible singleton type; thus this
        method relies on helper functions to obtain
        token ids for the sub-prompts.

        Arguments:

        * prompt: an input prompt

        Returns:

        * [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
          instance
        """
        logger.info(f"[SxlAdd] 开始处理encoder-decoder模型提示")
        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs | None
        if is_explicit_encoder_decoder_prompt(prompt):
            # `cast` is needed for mypy, but not pyright
            prompt_ = cast(ExplicitEncoderDecoderPrompt, prompt)
            logger.info(f"[SxlAdd] 处理显式encoder-decoder提示")
            encoder_inputs = self._prompt_to_llm_inputs(
                prompt_["encoder_prompt"],
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            logger.info(f"[SxlAdd] 编码器提示处理完成")
            if (decoder_input := prompt_["decoder_prompt"]) is None:
                decoder_inputs = None
                logger.info(f"[SxlAdd] 解码器提示为None")
            else:
                decoder_inputs = self._prompt_to_llm_inputs(
                    decoder_input, tokenization_kwargs=tokenization_kwargs
                )
                logger.info(f"[SxlAdd] 解码器提示处理完成")
            # For multimodal model, override decoder prompt from processor
            # with explicit decoder prompt.
            if self.model_config.is_multimodal_model:
                logger.info(f"[SxlAdd] 处理多模态encoder-decoder模型")
                encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(
                    encoder_inputs, decoder_inputs
                )
                logger.info(f"[SxlAdd] 多模态输入分离完成")
        else:
            # `cast` is needed for mypy, but not pyright
            logger.info(f"[SxlAdd] 处理单例提示")
            inputs = self._prompt_to_llm_inputs(
                cast(SingletonPrompt, prompt),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            logger.info(f"[SxlAdd] 单例提示处理完成")
            if self.model_config.is_multimodal_model:
                # Encoder-Decoder Multimodal model
                logger.info(f"[SxlAdd] 处理多模态encoder-decoder模型")
                encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(inputs)
                logger.info(f"[SxlAdd] 多模态输入分离完成")
            else:
                encoder_inputs = inputs
                decoder_inputs = None

        result = self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)
        logger.info(f"[SxlAdd] 构建encoder-decoder输入完成")
        return result

    def _build_decoder_only_llm_inputs(
        self,
        prompt_inputs: DecoderOnlyInputs,
    ) -> DecoderOnlyInputs:
        if "prompt_token_ids" in prompt_inputs:
            prompt_inputs = cast(
                TokenInputs | MultiModalInputs, prompt_inputs
            )  # Needed for mypy

        return prompt_inputs

    def _process_decoder_only_prompt(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderOnlyInputs:
        """
        For decoder-only models:
        Process an input prompt into a
        [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance.

        Arguments:

        * prompt: input prompt

        Returns:

        * [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance
        """
        logger.info(f"[SxlAdd] 开始处理decoder-only模型提示")
        prompt_comps = self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        logger.info(f"[SxlAdd] 提示转换完成，结果类型: {type(prompt_comps).__name__}")
        result = self._build_decoder_only_llm_inputs(prompt_comps)
        logger.info(f"[SxlAdd] 构建decoder-only输入完成")
        return result

    def _preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        logger.info(f"[SxlAdd] 开始预处理，模型类型: {'encoder-decoder' if self.model_config.is_encoder_decoder else 'decoder-only'}")
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            logger.info(f"[SxlAdd] 处理encoder-decoder模型提示")
            return self._process_encoder_decoder_prompt(
                prompt,
                tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        if is_explicit_encoder_decoder_prompt(prompt):
            raise ValueError(
                "Cannot pass encoder-decoder prompt to decoder-only models"
            )

        # Decoder-only operation
        # `cast` is needed for mypy, but not pyright
        logger.info(f"[SxlAdd] 处理decoder-only模型提示")
        return self._process_decoder_only_prompt(
            cast(SingletonPrompt, prompt),
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        logger.info(f"[SxlAdd] 开始处理输入提示，提示类型: {type(prompt).__name__}")
        res = self._preprocess(prompt, tokenization_kwargs, mm_uuids=mm_uuids)
        logger.info(f"[SxlAdd] 输入提示处理完成，返回类型: {type(res).__name__}")

        if self.mm_processor_cache and self.mm_cache_stats is not None:
            delta = self.mm_processor_cache.make_stats(delta=True)
            self.mm_cache_stats.requests += 1
            self.mm_cache_stats.queries += delta.total
            self.mm_cache_stats.hits += delta.hits
            logger.info(f"[SxlAdd] 多模态缓存统计: 请求数={self.mm_cache_stats.requests}, 查询数={delta.total}, 命中数={delta.hits}")

        return res

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        mm_cache_stats = self.mm_cache_stats
        if mm_cache_stats is None:
            return None

        self.mm_cache_stats = MultiModalCacheStats()

        return mm_cache_stats

    def clear_mm_cache(self) -> None:
        if self.mm_processor_cache is not None:
            self.mm_processor_cache.clear_cache()

        if self.mm_cache_stats is not None:
            self.mm_cache_stats.reset = True
