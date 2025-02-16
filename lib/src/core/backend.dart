import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';

class BackendDevice extends LLAMAClass<C.ggml_backend_device> {
  BackendDevice(super.ptr);

  @override
  void dispose() {}
}

class BackendBuffer extends LLAMAClass<C.ggml_backend_buffer> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  BackendBuffer(super.ptr);

  @override
  void dispose() {}
}

/// Initialize the llama + ggml backend
/// If numa is true, use NUMA optimizations
/// Call once at the start of the program
void backendInit() => C.llama_backend_init();

/// Call once at the end of the program - currently only used for MPI
void backendFree() => C.llama_backend_free();
