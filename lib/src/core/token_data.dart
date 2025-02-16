import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';

class TokenData extends LLAMAStruct<C.llama_token_data> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  TokenData(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  /// token id
  int get id => ref.id;

  /// token id
  set id(int value) => ref.id = value;

  /// log-odds of the token
  double get logit => ref.logit;

  /// log-odds of the token
  set logit(double value) => ref.logit = value;

  /// probability of the token
  double get p => ref.p;

  /// probability of the token
  set p(double value) => ref.p = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  C.llama_token_data get ref => ptr.ref;
}

class TokenDataArray extends LLAMAStruct<C.llama_token_data_array> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  TokenDataArray(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  /// NOTE: this pointer can be modified by the samplers
  ffi.Pointer<C.llama_token_data> get data => ref.data;

  int get size => ref.size;
  set size(int value) => ref.size = value;

  /// this is the index in the data array (i.e. not the token id)
  int get selected => ref.selected;

  /// this is the index in the data array (i.e. not the token id)
  set selected(int value) => ref.selected = value;

  bool get sorted => ref.sorted;
  set sorted(bool value) => ref.sorted = value;

  @override
  void dispose() {
    finalizer.detach(this);
    // calloc.free(ptr.ref.data);
    calloc.free(ptr);
  }

  @override
  C.llama_token_data_array get ref => ptr.ref;
}
