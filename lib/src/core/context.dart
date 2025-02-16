import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'adapter.dart';
import 'base.dart';
import 'batch.dart';
import 'enums.dart';
import 'exception.dart';
import 'model.dart';

/// NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes
/// or incorrect results in certain configurations
/// https://github.com/ggerganov/llama.cpp/pull/7544
class ContextParams extends LLAMAStruct<C.llama_context_params> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ContextParams(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory ContextParams.create() {
    final params = C.llama_context_default_params();
    final p = calloc<C.llama_context_params>()..ref = params;
    return ContextParams(p);
  }

  /// text context, 0 = from model
  int get nCtx => ref.n_ctx;
  set nCtx(int value) => ref.n_ctx = value;

  /// logical maximum batch size that can be submitted to llama_decode
  int get nBatch => ref.n_batch;
  set nBatch(int value) => ref.n_batch = value;

  /// physical maximum batch size
  int get nUbatch => ref.n_ubatch;
  set nUbatch(int value) => ref.n_ubatch = value;

  /// max number of sequences (i.e. distinct states for recurrentmodels)
  int get nSeqMax => ref.n_seq_max;
  set nSeqMax(int value) => ref.n_seq_max = value;

  /// number of threads to use for generation
  int get nThreads => ref.n_threads;
  set nThreads(int value) => ref.n_threads = value;

  /// number of threads to use for batch processing
  int get nThreadsBatch => ref.n_threads_batch;
  set nThreadsBatch(int value) => ref.n_threads_batch = value;

  /// RoPE scaling type, from `enum llama_rope_scaling_type`
  RopeScalingType get ropeScalingType => ref.rope_scaling_type;
  set ropeScalingType(RopeScalingType value) => ref.rope_scaling_typeAsInt = value.value;

  /// whether to pool (sum) embedding results by sequence id
  PoolingType get poolingType => ref.pooling_type;
  set poolingType(PoolingType value) => ref.pooling_typeAsInt = value.value;

  /// attention type to use for embeddings
  AttentionType get attentionType => ref.attention_type;
  set attentionType(AttentionType value) => ref.attention_typeAsInt = value.value;

  /// RoPE base frequency, 0 = from model
  double get ropeFreqBase => ref.rope_freq_base;
  set ropeFreqBase(double value) => ref.rope_freq_base = value;

  /// RoPE frequency scaling factor, 0 = from model
  double get ropeFreqScale => ref.rope_freq_scale;
  set ropeFreqScale(double value) => ref.rope_freq_scale = value;

  /// YaRN extrapolation mix factor, negative = from model
  double get yarnExtFactor => ref.yarn_ext_factor;
  set yarnExtFactor(double value) => ref.yarn_ext_factor = value;

  /// YaRN attention mix factor, negative = from model
  double get yarnAttnFactor => ref.yarn_attn_factor;
  set yarnAttnFactor(double value) => ref.yarn_attn_factor = value;

  /// YaRN low correction dim, negative = from model
  double get yarnBetaFast => ref.yarn_beta_fast;
  set yarnBetaFast(double value) => ref.yarn_beta_fast = value;

  /// YaRN high correction dim, negative = from model
  double get yarnBetaSlow => ref.yarn_beta_slow;
  set yarnBetaSlow(double value) => ref.yarn_beta_slow = value;

  /// YaRN original context size, negative = from model
  int get yarnOrigCtx => ref.yarn_orig_ctx;
  set yarnOrigCtx(int value) => ref.yarn_orig_ctx = value;

  /// defragment the KV cache if holes/size > thold, < 0 disabled (default)
  double get defragThold => ref.defrag_thold;
  set defragThold(double value) => ref.defrag_thold = value;

// TODO: add support for ggml_backend_sched_eval_callback
//   external ggml_backend_sched_eval_callback cb_eval;

//   external ffi.Pointer<ffi.Void> cb_eval_user_data;

  /// data type for K cache [EXPERIMENTAL]
  GGMLType get typeK => ref.type_k;
  set typeK(GGMLType value) => ref.type_kAsInt = value.value;

  /// data type for V cache [EXPERIMENTAL]
  GGMLType get typeV => ref.type_v;
  set typeV(GGMLType value) => ref.type_vAsInt = value.value;

  /// the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
  bool get logitsAll => ref.logits_all;
  set logitsAll(bool value) => ref.logits_all = value;

  /// if true, extract embeddings (together with logits)
  bool get embeddings => ref.embeddings;
  set embeddings(bool value) => ref.embeddings = value;

  /// whether to offload the KQV ops (including the KV cache) to GPU
  bool get offloadKQV => ref.offload_kqv;
  set offloadKQV(bool value) => ref.offload_kqv = value;

  /// whether to use flash attention [EXPERIMENTAL]
  bool get flashAttn => ref.flash_attn;
  set flashAttn(bool value) => ref.flash_attn = value;

  /// whether to measure performance timings
  bool get noPerf => ref.no_perf;
  set noPerf(bool value) => ref.no_perf = value;

  /// Abort callback
  /// if it returns true, execution of llama_decode() will be aborted
  /// currently works only with CPU execution
  // external ggml_abort_callback abort_callback;

  // external ffi.Pointer<ffi.Void> abort_callback_data;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.llama_context_params get ref => ptr.ref;
}

class Context extends LLAMAClass<C.llama_context> {
  static final finalizer = ffi.NativeFinalizer(C.addresses.llama_free.cast());
  Context(super.ptr, {bool attach = true}) {
    if (ptr == ffi.nullptr) {
      throw LLAMAException('Model not loaded');
    }
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory Context.fromModel(Model model, {ContextParams? params}) {
    params ??= ContextParams.create();
    final ptr = C.llama_init_from_model(model.ptr, params.ref);
    if (ptr == ffi.nullptr) {
      model.dispose();
      throw LLAMAException('Failed to create context');
    }
    return Context(ptr);
  }

  int get nCtx => C.llama_n_ctx(ptr);

  int get nBatch => C.llama_n_batch(ptr);

  int get nUbatch => C.llama_n_ubatch(ptr);

  int get nSeqMax => C.llama_n_seq_max(ptr);

  Model get model => Model(C.llama_get_model(ptr)); // TODO: attach?

  PoolingType get poolingType => C.llama_pooling_type(ptr);

  /// Add a loaded LoRA adapter to given context
  /// This will not modify model's weight
  int setAdapterLoRA(AdapterLoRA adapter, double scale) => C.llama_set_adapter_lora(ptr, adapter.ptr, scale);

  /// Remove a specific LoRA adapter from given context
  /// Return -1 if the adapter is not present in the context
  int rmAdapterLoRA(AdapterLoRA adapter) => C.llama_rm_adapter_lora(ptr, adapter.ptr);

  /// Remove all LoRA adapters from given context
  void clearAdapterLoRA() => C.llama_clear_adapter_lora(ptr);

  /// Apply a loaded control vector to a llama_context, or if data is NULL, clear
  /// the currently loaded vector.
  /// n_embd should be the size of a single layer's control, and data should point
  /// to an n_embd x n_layers buffer starting from layer 1.
  /// il_start and il_end are the layer range the vector should apply to (both inclusive)
  /// See llama_control_vector_load in common to load a control vector.
  int applyAdapterCVec(Float32List data, int len, int nEmbd, int ilStart, int ilEnd) {
    return C.llama_apply_adapter_cvec(ptr, data.address, len, nEmbd, ilStart, ilEnd);
  }

  /// Returns the number of tokens in the KV cache (slow, use only for debug)
  /// If a KV cell has multiple sequences assigned to it, it will be counted multiple times
  int get kvCacheTokenCount => C.llama_get_kv_cache_token_count(ptr);

  /// Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
  int get kvCacheUsedCells => C.llama_get_kv_cache_used_cells(ptr);

  /// Clear the KV cache - both cell info is erased and KV data is zeroed
  void kvCacheClear() => C.llama_kv_cache_clear(ptr);

  /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
  /// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
  /// seq_id < 0 : match any sequence
  /// p0 < 0     : [0,  p1]
  /// p1 < 0     : [p0, inf)
  bool kvCacheSeqRM(int seqId, int p0, int p1) => C.llama_kv_cache_seq_rm(ptr, seqId, p0, p1);

  /// Copy all tokens that belong to the specified sequence to another sequence
  /// Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
  /// p0 < 0 : [0,  p1]
  /// p1 < 0 : [p0, inf)
  void kvCacheSeqCP(int seqIdSrc, int seqIdDst, int p0, int p1) =>
      C.llama_kv_cache_seq_cp(ptr, seqIdSrc, seqIdDst, p0, p1);

  /// Removes all tokens that do not belong to the specified sequence
  void kvCacheSeqKeep(int seqId) => C.llama_kv_cache_seq_keep(ptr, seqId);

  /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
  /// If the KV cache is RoPEd, the KV data is updated accordingly:
  /// - lazily on next llama_decode()
  /// - explicitly with llama_kv_cache_update()
  /// p0 < 0 : [0,  p1]
  /// p1 < 0 : [p0, inf)
  void kvCacheSeqAdd(int seqId, int p0, int p1, int delta) =>
      C.llama_kv_cache_seq_add(ptr, seqId, p0, p1, delta);

  /// Integer division of the positions by factor of `d > 1`
  /// If the KV cache is RoPEd, the KV data is updated accordingly:
  /// - lazily on next llama_decode()
  /// - explicitly with llama_kv_cache_update()
  /// p0 < 0 : [0,  p1]
  /// p1 < 0 : [p0, inf)
  void kvCacheSeqDiv(int seqId, int p0, int p1, int d) => C.llama_kv_cache_seq_div(ptr, seqId, p0, p1, d);

  /// Returns the largest position present in the KV cache for the specified sequence
  int kvCacheSeqPosMax(int seqId) => C.llama_kv_cache_seq_pos_max(ptr, seqId);

  /// Defragment the KV cache
  /// This will be applied:
  /// - lazily on next llama_decode()
  /// - explicitly with llama_kv_cache_update()
  void kvCacheDefrag() => C.llama_kv_cache_defrag(ptr);

  /// Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
  void kvCacheUpdate() => C.llama_kv_cache_update(ptr);

  /// Check if the context supports KV cache shifting
  bool kvCacheCanShift() => C.llama_kv_cache_can_shift(ptr);

  /// Returns the *actual* size in bytes of the state
  /// (logits, embedding and kv_cache)
  /// Only use when saving the state, not when restoring it, otherwise the size may be too small.
  int get stateSize => C.llama_state_get_size(ptr);

  /// Copies the state to the specified destination address.
  /// Destination needs to have allocated enough memory.
  /// Returns the number of bytes copied
  (int nBytes, Uint8List data) getState(int size) {
    final p = calloc<ffi.Uint8>(size);
    final nBytes = C.llama_state_get_data(ptr, p, size);
    return (nBytes, p.asTypedList(size));
  }

  /// Set the state reading from the specified address
  /// Returns the number of bytes read
  int setState(Uint8List data) {
    final nBytes = C.llama_state_set_data(ptr, data.address, data.length);
    return nBytes;
  }

  /// load session file
  (bool success, Int32List tokens) loadStateFromFile(String pathSession, {int nTokenCapacity = 1024}) {
    final cpath = pathSession.toNativeUtf8().cast<ffi.Char>();
    final pTokensOut = calloc<C.llama_token>(nTokenCapacity);
    final pCount = calloc<ffi.Size>();
    final result = C.llama_state_load_file(ptr, cpath, pTokensOut, nTokenCapacity, pCount);
    final nTokenCount = pCount.value;
    calloc.free(cpath);
    calloc.free(pCount);
    return (result, pTokensOut.asTypedList(nTokenCount));
  }

  /// save session file
  bool saveStateToFile(String pathSession, Int32List tokens) {
    final cpath = pathSession.toNativeUtf8().cast<ffi.Char>();
    final result = C.llama_state_save_file(ptr, cpath, tokens.address, tokens.length);
    calloc.free(cpath);
    return result;
  }

  /// Get the exact size needed to copy the KV cache of a single sequence
  int getStateSeqSize(int seqId) => C.llama_state_seq_get_size(ptr, seqId);

  /// Copy the KV cache of a single sequence into the specified buffer
  (int, Uint8List dst) getStateSeqData(int size, int seqId) {
    final p = calloc<ffi.Uint8>(size);
    final nBytes = C.llama_state_seq_get_data(ptr, p, size, seqId);
    return (nBytes, p.asTypedList(size));
  }

  /// Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
  /// Returns:
  /// - Positive: Ok
  /// - Zero: Failed to load
  // external int llama_state_seq_set_data(
  //   ffi.Pointer<llama_context> ctx,
  //   ffi.Pointer<ffi.Uint8> src,
  //   int size,
  //   int dest_seq_id,
  // );
  int setStateSeqData(Uint8List data, int seqId) {
    final result = C.llama_state_seq_set_data(ptr, data.address, data.length, seqId);
    return result;
  }

  int saveStateSeqToFile(String filepath, int seqId, Int32List tokens) {
    final cpath = filepath.toNativeUtf8().cast<ffi.Char>();
    final result = C.llama_state_seq_save_file(ptr, cpath, seqId, tokens.address, tokens.length);
    calloc.free(cpath);
    return result;
  }

  (int success, Int32List tokens) loadStateSeqFromFile(
    String filepath,
    int destSeqId, {
    int nTokenCapacity = 1024,
  }) {
    final cpath = filepath.toNativeUtf8().cast<ffi.Char>();
    final pTokensOut = calloc<C.llama_token>(nTokenCapacity);
    final pCount = calloc<ffi.Size>();
    final result = C.llama_state_seq_load_file(ptr, cpath, destSeqId, pTokensOut, nTokenCapacity, pCount);
    final nTokenCount = pCount.value;
    calloc.free(cpath);
    calloc.free(pCount);
    return (result, pTokensOut.cast<ffi.Int32>().asTypedList(nTokenCount));
  }

  /// Return batch for single sequence of tokens
  /// The sequence ID will be fixed to 0
  /// The position of the tokens will be tracked automatically by llama_decode
  ///
  /// NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
  Batch getOneBatch(Int32List tokens) {
    final p = C.llama_batch_get_one(tokens.address, tokens.length);
    return Batch.fromNative(p);
  }

  /// Processes a batch of tokens with the ecoder part of the encoder-decoder model.
  /// Stores the encoder output internally for later use by the decoder cross-attention layers.
  /// 0 - success
  /// < 0 - error. the KV cache state is restored to the state before this call
  int encode(Batch batch) => C.llama_encode(ptr, batch.ref);

  /// Positive return values does not mean a fatal error, but rather a warning.
  /// 0 - success
  /// 1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
  /// < 0 - error. the KV cache state is restored to the state before this call
  int decode(Batch batch) => C.llama_decode(ptr, batch.ref);

  /// Set the number of threads used for decoding
  /// n_threads is the number of threads used for generation (single token)
  /// n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
  void setNThreads(int nThreads, int nThreadsBatch) => C.llama_set_n_threads(ptr, nThreads, nThreadsBatch);

  /// Get the number of threads used for generation of a single token.
  int get nThreads => C.llama_n_threads(ptr);

  /// Get the number of threads used for prompt and batch processing (multiple token).
  int get nThreadsBatch => C.llama_n_threads_batch(ptr);

  /// Set whether the model is in embeddings mode or not
  /// If true, embeddings will be returned but logits will not
  void setEmbeddings(bool embeddings) => C.llama_set_embeddings(ptr, embeddings);

  /// Set whether to use causal attention or not
  /// If set to true, the model will only attend to the past tokens
  void setCausalAttn(bool causalAttn) => C.llama_set_causal_attn(ptr, causalAttn);

  /// Set abort callback
  /// TODO
  // external void llama_set_abort_callback(
  //   ffi.Pointer<llama_context> ctx,
  //   ggml_abort_callback abort_callback,
  //   ffi.Pointer<ffi.Void> abort_callback_data,
  // );

  /// Wait until all computations are finished
  /// This is automatically done when using one of the functions below to obtain the computation results
  /// and is not necessary to call it explicitly in most cases
  void synchronize() => C.llama_synchronize(ptr);

  /// Token logits obtained from the last call to llama_decode()
  /// The logits for which llama_batch.logits[i] != 0 are stored contiguously
  /// in the order they have appeared in the batch.
  /// Rows: number of tokens for which llama_batch.logits[i] != 0
  /// Cols: n_vocab
  ffi.Pointer<ffi.Float> getLogits() => C.llama_get_logits(ptr);

  /// Logits for the ith token. For positive indices, Equivalent to:
  /// llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
  /// Negative indicies can be used to access logits in reverse order, -1 is the last logit.
  /// returns NULL for invalid ids.
  ffi.Pointer<ffi.Float> getLogitsIth(int i) {
    final p = C.llama_get_logits_ith(ptr, i);
    return p;
  }

  /// Get all output token embeddings.
  /// when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
  /// the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
  /// in the order they have appeared in the batch.
  /// shape: [n_outputs*n_embd]
  /// Otherwise, returns NULL.
  // external ffi.Pointer<ffi.Float> llama_get_embeddings(
  //   ffi.Pointer<llama_context> ctx,
  // );
  ffi.Pointer<ffi.Float> getEmbeddings() => C.llama_get_embeddings(ptr);

  /// Get the embeddings for the ith token. For positive indices, Equivalent to:
  /// llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
  /// Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
  /// shape: [n_embd] (1-dimensional)
  /// returns NULL for invalid ids.
  // external ffi.Pointer<ffi.Float> llama_get_embeddings_ith(
  //   ffi.Pointer<llama_context> ctx,
  //   int i,
  // );
  ffi.Pointer<ffi.Float> getEmbeddingsIth(int i) {
    return C.llama_get_embeddings_ith(ptr, i);
  }

  /// Get the embeddings for a sequence id
  /// Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
  /// when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
  /// otherwise: float[n_embd] (1-dimensional)
  ffi.Pointer<ffi.Float> getEmbeddingsSeq(int seqId) {
    return C.llama_get_embeddings_seq(ptr, seqId);
  }

  @override
  void dispose() {
    finalizer.detach(this);
    C.llama_free(ptr);
  }
}
