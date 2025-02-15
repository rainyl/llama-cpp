import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:llama_cpp/src/core/model_params.dart';
import 'package:llama_cpp/src/core/quantize.dart';

import 'base.dart';
import 'enums.dart';
import '../g/llama.g.dart' as llama;
import 'vocab.dart';

class Model extends LLAMAClass<llama.llama_model> {
  static final finalizer = ffi.NativeFinalizer(llama.addresses.llama_model_free.cast());

  Model(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  /// Load the model from a file
  /// If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
  /// If the split file name does not follow this pattern, use llama_model_load_from_splits
  factory Model.fromFile(String path, {ModelParams? params}) {
    params ??= ModelParams.create();
    final cpath = path.toNativeUtf8().cast<ffi.Char>();
    final ptr = llama.llama_model_load_from_file(cpath, params.ref);
    calloc.free(cpath);
    return Model(ptr);
  }

  /// Load the model from multiple splits (support custom naming scheme)
  /// The paths must be in the correct order
  factory Model.fromSplits(List<String> paths, {ModelParams? params}) {
    params ??= ModelParams.create();
    final cpaths = calloc<ffi.Pointer<ffi.Char>>(paths.length);
    for (var i = 0; i < paths.length; i++) {
      cpaths[i] = paths[i].toNativeUtf8().cast<ffi.Char>();
    }
    final ptr = llama.llama_model_load_from_splits(cpaths, paths.length, params.ref);
    for (var i = 0; i < paths.length; i++) {
      calloc.free(cpaths[i]);
    }
    calloc.free(cpaths);
    return Model(ptr);
  }

  Vocab get vocab => Vocab(llama.llama_model_get_vocab(ptr));

  RopeType get ropeType => llama.llama_model_rope_type(ptr);

  int get nCtxTrain => llama.llama_model_n_ctx_train(ptr);

  int get nEmbd => llama.llama_model_n_embd(ptr);

  int get nLayer => llama.llama_model_n_layer(ptr);

  int get nHead => llama.llama_model_n_head(ptr);

  /// Get the model's RoPE frequency scaling factor
  double get ropeFreqScaleTrain => llama.llama_model_rope_freq_scale_train(ptr);

  /// Get metadata value as a string by key name
  /// TODO
  // external int llama_model_meta_val_str(
  //   ffi.Pointer<llama_model> model,
  //   ffi.Pointer<ffi.Char> key,
  //   ffi.Pointer<ffi.Char> buf,
  //   int buf_size,
  // );

  /// Get the number of metadata key/value pairs
  int get metaCount => llama.llama_model_meta_count(ptr);

  /// Get metadata key name by index
  /// TODO
  // external int llama_model_meta_key_by_index(
  //   ffi.Pointer<llama_model> model,
  //   int i,
  //   ffi.Pointer<ffi.Char> buf,
  //   int buf_size,
  // );

  /// Get metadata value as a string by index
  /// TODO
  // external int llama_model_meta_val_str_by_index(
  //   ffi.Pointer<llama_model> model,
  //   int i,
  //   ffi.Pointer<ffi.Char> buf,
  //   int buf_size,
  // );

  /// Get a string describing the model type
  // external int llama_model_desc(
  //   ffi.Pointer<llama_model> model,
  //   ffi.Pointer<ffi.Char> buf,
  //   int buf_size,
  // );

  /// Returns the total size of all the tensors in the model in bytes
  int get size => llama.llama_model_size(ptr);

  /// Get the default chat template. Returns nullptr if not available
  /// If name is NULL, returns the default chat template
  // external ffi.Pointer<ffi.Char> llama_model_chat_template(
  //   ffi.Pointer<llama_model> model,
  //   ffi.Pointer<ffi.Char> name,
  // );
  String getChatTemplate({String? name}) {
    final cname = name?.toNativeUtf8().cast<ffi.Char>() ?? ffi.nullptr;
    final p = llama.llama_model_chat_template(ptr, cname);
    if (cname != ffi.nullptr) calloc.free(cname);
    final rval = p.cast<Utf8>().toDartString();
    calloc.free(p);
    return rval;
  }

  /// Returns the total number of parameters in the model
  int get nParams => llama.llama_model_n_params(ptr);

  /// Returns true if the model contains an encoder that requires llama_encode() call
  bool get hasEncoder => llama.llama_model_has_encoder(ptr);

  /// Returns true if the model contains a decoder that requires llama_decode() call
  bool get hasDecoder => llama.llama_model_has_decoder(ptr);

  /// For encoder-decoder models, this function returns id of the token that must be provided
  /// to the decoder to start generating output sequence. For other models, it returns -1.
  int get decoderStartToken => llama.llama_model_decoder_start_token(ptr);

  /// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
  bool get isRecurrent => llama.llama_model_is_recurrent(ptr);

  static int quantize(String path, String pathOut, {ModelQuantizeParams? params}) {
    params ??= ModelQuantizeParams.create();
    final cpath = path.toNativeUtf8().cast<ffi.Char>();
    final cpathOut = pathOut.toNativeUtf8().cast<ffi.Char>();
    final rval = llama.llama_model_quantize(cpath, cpathOut, params.ptr);
    calloc.free(cpath);
    calloc.free(cpathOut);
    return rval;
  }

  @override
  void dispose() {
    finalizer.detach(this);
    llama.llama_model_free(ptr);
  }
}
