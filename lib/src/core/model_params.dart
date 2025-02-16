import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';
import 'enums.dart';

class ModelParams extends LLAMAStruct<C.llama_model_params> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ModelParams(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory ModelParams.create() {
    final params = C.llama_model_default_params();
    final ptr = calloc<C.llama_model_params>()..ref = params;
    // calloc.free(params.address);
    return ModelParams(ptr);
  }

  /// NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
  // ffi.Pointer<ggml_backend_dev_t> devices;

  /// number of layers to store in VRAM
  int get nGPULayers => ref.n_gpu_layers;

  /// number of layers to store in VRAM
  set nGPULayers(int value) => ref.n_gpu_layers = value;

  /// how to split the model across multiple GPUs
  SplitMode get splitMode => ref.split_mode;
  set splitMode(SplitMode value) => ref.split_modeAsInt = value.value;

  /// the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
  int get mainGPU => ref.main_gpu;
  set mainGPU(int value) => ref.main_gpu = value;

  /// proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
  /// TODO: setter?
  ffi.Pointer<ffi.Float> get tensorSplit => ref.tensor_split;

  /// Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
  /// If the provided progress_callback returns true, model loading continues.
  /// If it returns false, model loading is immediately aborted.
  // llama_progress_callback progress_callback;

  /// context pointer passed to the progress callback
  // ffi.Pointer<ffi.Void> progress_callback_user_data;

  /// override key-value pairs of the model meta data
  // ffi.Pointer<llama_model_kv_override> kv_overrides;

  /// only load the vocabulary, no weights
  bool get vocabOnly => ref.vocab_only;
  set vocabOnly(bool value) => ref.vocab_only = value;

  /// use mmap if possible
  bool get useMmap => ref.use_mmap;
  set useMmap(bool value) => ref.use_mmap = value;

  /// force system to keep model in RAM
  bool get useMlock => ref.use_mlock;
  set useMlock(bool value) => ref.use_mlock = value;

  /// validate model tensor data
  bool get checkTensors => ref.check_tensors;
  set checkTensors(bool value) => ref.check_tensors = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.llama_model_params get ref => ptr.ref;
}

class SamplerChainParams extends LLAMAStruct<C.llama_sampler_chain_params> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  SamplerChainParams(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory SamplerChainParams.create() {
    final params = C.llama_sampler_chain_default_params();
    final ptr = calloc<C.llama_sampler_chain_params>()..ref = params;
    // calloc.free(params.address);
    return SamplerChainParams(ptr);
  }

  /// whether to measure performance timings
  bool get noPerf => ref.no_perf;
  set noPerf(bool value) => ref.no_perf = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.llama_sampler_chain_params get ref => ptr.ref;
}
