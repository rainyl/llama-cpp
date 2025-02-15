import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import '../g/llama.g.dart' as llama;

class BackendDevice extends LLAMAClass<llama.ggml_backend_device> {
  BackendDevice(super.ptr, {bool attach = true});

  @override
  void dispose() {}
}

class BackendBuffer extends LLAMAClass<llama.ggml_backend_buffer> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  BackendBuffer(super.ptr);

  @override
  void dispose() {}
}

/// Initialize the llama + ggml backend
/// If numa is true, use NUMA optimizations
/// Call once at the start of the program
void backendInit() => llama.llama_backend_init();

/// Call once at the end of the program - currently only used for MPI
void backendFree() => llama.llama_backend_free();
