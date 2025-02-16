import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';
import 'model.dart';

class AdapterLoRA extends LLAMAClass<C.llama_adapter_lora> {
  AdapterLoRA(super.ptr);

  /// Load a LoRA adapter from file
  factory AdapterLoRA.init(Model model, String path) {
    final cpath = path.toNativeUtf8().cast<ffi.Char>();
    final ptr = C.llama_adapter_lora_init(model.ptr, cpath);
    calloc.free(cpath);
    return AdapterLoRA(ptr);
  }

  /// Manually free a LoRA adapter
  /// Note: loaded adapters will be free when the associated model is deleted
  @override
  void dispose() {
    C.llama_adapter_lora_free(ptr);
  }
}
