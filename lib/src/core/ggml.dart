import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';
import 'enums.dart';

/// n-dimensional tensor
class Tensor extends LLAMAStruct<C.ggml_tensor> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  Tensor(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  GGMLType get type => ref.type;
  set type(GGMLType value) => ref.typeAsInt = value.value;

//   external ffi.Pointer<ggml_backend_buffer> buffer;

  /// number of elements
  @ffi.Array.multi([4])
  ffi.Array<ffi.Int64> get ne => ref.ne;

  /// stride in bytes:
  /// nb[0] = ggml_type_size(type)
  /// nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
  /// nb[i] = nb[i-1] * ne[i-1]
  @ffi.Array.multi([4])
  ffi.Array<ffi.Size> get nb => ref.nb;

  GGMOp get op => ref.op;
  set op(GGMOp value) => ref.opAsInt = value.value;

  /// compute data
  /// op params - allocated as int32_t for alignment
  @ffi.Array.multi([16])
  ffi.Array<ffi.Int32> get opParams => ref.op_params;

  int get flags => ref.flags;
  set flags(int value) => ref.flags = value;

  // @ffi.Array.multi([10])
  // ffi.Array<ffi.Pointer<llama.ggml_tensor>> src;

  /// source tensor and offset for views
  // ffi.Pointer<ggml_tensor> view_src;

  int get viewOffs => ref.view_offs;

  // ffi.Pointer<ffi.Void> data;

  // @ffi.Array.multi([64])
  // ffi.Array<ffi.Char> name;
  String get name => String.fromCharCodes(List.generate(64, (i) => ref.name[i]));

  /// extra things e.g. for ggml-cuda.cu
  // ffi.Pointer<ffi.Void> extra;

  // @ffi.Array.multi([8])
  // ffi.Array<ffi.Char> padding;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.ggml_tensor get ref => ptr.ref;
}
