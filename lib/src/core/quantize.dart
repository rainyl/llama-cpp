import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';
import 'enums.dart';

class ModelQuantizeParams extends LLAMAStruct<C.llama_model_quantize_params> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ModelQuantizeParams(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory ModelQuantizeParams.create() {
    final params = C.llama_model_quantize_default_params();
    final p = calloc<C.llama_model_quantize_params>()..ref = params;
    return ModelQuantizeParams(p);
  }

  /// number of threads to use for quantizing
  int get nthread => ref.nthread;
  set nthread(int value) => ref.nthread = value;

  /// quantize to this llama_ftype
  FileType get ftype => ref.ftype;
  set ftype(FileType value) => ref.ftypeAsInt = value.value;

  /// output tensor type
  GGMLType get outputTensorType => ref.output_tensor_type;
  set outputTensorType(GGMLType value) => ref.output_tensor_typeAsInt = value.value;

  /// token embeddings tensor type
  GGMLType get tokenEmbeddingType => ref.token_embedding_type;
  set tokenEmbeddingType(GGMLType value) => ref.token_embedding_typeAsInt = value.value;

  /// allow quantizing non-f32/f16 tensors
  bool get allowRequantize => ref.allow_requantize;
  set allowRequantize(bool value) => ref.allow_requantize = value;

  /// quantize output.weight
  bool get quantizeOutputTensor => ref.quantize_output_tensor;
  set quantizeOutputTensor(bool value) => ref.quantize_output_tensor = value;

  /// only copy tensors
  bool get onlyCopy => ref.only_copy;
  set onlyCopy(bool value) => ref.only_copy = value;

  /// quantize all tensors to the default type
  bool get pure => ref.pure;
  set pure(bool value) => ref.pure = value;

  /// quantize to the same number of shards
  bool get keepSplit => ref.keep_split;
  set keepSplit(bool value) => ref.keep_split = value;

  /// pointer to importance matrix data
  ffi.Pointer<ffi.Void> get imatrix => ref.imatrix;
  set imatrix(ffi.Pointer<ffi.Void> value) => ref.imatrix = value;

  /// pointer to vector containing overrides
  ffi.Pointer<ffi.Void> get kvOverrides => ref.kv_overrides;
  set kvOverrides(ffi.Pointer<ffi.Void> value) => ref.kv_overrides = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.llama_model_quantize_params get ref => ptr.ref;
}
