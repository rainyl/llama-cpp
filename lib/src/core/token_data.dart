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

  factory TokenData.create({int? id, double? logit, double? p}) {
    final ptr = calloc<C.llama_token_data>();
    if (id != null) ptr.ref.id = id;
    if (logit != null) ptr.ref.logit = logit;
    if (p != null) ptr.ref.p = p;
    return TokenData(ptr);
  }

  factory TokenData.fromNative(C.llama_token_data struct) {
    final ptr = calloc<C.llama_token_data>()..ref = struct;
    return TokenData(ptr);
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

  @override
  String toString() =>
      'TokenData(address=0x${ptr.address.toRadixString(16)}, id=$id, logit=${logit.toStringAsFixed(3)}, p=${p.toStringAsFixed(3)})';
}

class TokenDataArray extends LLAMAStruct<C.llama_token_data_array> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  TokenDataArray(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory TokenDataArray.fromList(
    List<TokenData> list, {
    int selected = -1,
    bool sorted = false,
  }) {
    final ptr = calloc<C.llama_token_data_array>()
      ..ref.size = list.length
      ..ref.selected = selected
      ..ref.sorted = sorted;
    final pData = calloc<C.llama_token_data>(list.length);
    for (var i = 0; i < list.length; i++) {
      pData[i] = list[i].ref;
    }
    ptr.ref.data = pData;
    return TokenDataArray(ptr);
  }

  factory TokenDataArray.generate(
    int size,
    TokenData Function(int i) generator, {
    bool dispose = true,
    int selected = -1,
    bool sorted = false,
  }) {
    final ptr = calloc<C.llama_token_data_array>()
      ..ref.size = size
      ..ref.selected = selected
      ..ref.sorted = sorted;
    final pData = calloc<C.llama_token_data>(size);
    for (var i = 0; i < size; i++) {
      final tokenData = generator(i);
      pData[i] = tokenData.ref;
      if (dispose) tokenData.dispose();
    }
    ptr.ref.data = pData;
    return TokenDataArray(ptr);
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

  List<TokenData> toList() => List.generate(size, (i) => TokenData.fromNative(data[i]));

  @override
  void dispose() {
    finalizer.detach(this);
    // calloc.free(ptr.ref.data);
    calloc.free(ptr);
  }

  @override
  C.llama_token_data_array get ref => ptr.ref;

  @override
  String toString() =>
      'TokenDataArray(address=0x${ptr.address.toRadixString(16)}, size=$size, selected=$selected, sorted=$sorted)';
}
