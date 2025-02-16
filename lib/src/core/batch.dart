import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';

/// Input data for llama_decode
/// A llama_batch object can contain input about one or many sequences
/// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
///
/// - token  : the token ids of the input (used when embd is NULL)
/// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
/// - pos    : the positions of the respective token in the sequence
/// (if set to NULL, the token position will be tracked automatically by llama_decode)
/// - seq_id : the sequence to which the respective token belongs
/// (if set to NULL, the sequence ID will be assumed to be 0)
/// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
/// (if set to NULL, only the logits for last token will be returned)
class Batch extends LLAMAStruct<C.llama_batch> {
  static final finalizer = ffi.NativeFinalizer(C.addresses.llama_batch_free.cast());
  static final finalizer1 = ffi.NativeFinalizer(calloc.nativeFree);

  Batch(super.ptr, {this.numTokensAlloc = -1, bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
      finalizer1.attach(this, ptr.cast(), detach: this);
    }
  }

  /// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
  /// Each token can be assigned up to n_seq_max sequence ids
  /// The batch has to be freed with llama_batch_free()
  /// If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
  /// Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
  /// The rest of the llama_batch members are allocated with size n_tokens
  /// All members are left uninitialized
  factory Batch.init(int numTokensAlloc, int embd, int nSeqMax) {
    final b = C.llama_batch_init(numTokensAlloc, embd, nSeqMax);
    final p = calloc<C.llama_batch>()..ref = b;
    return Batch(p, numTokensAlloc: numTokensAlloc);
  }

  factory Batch.fromNative(C.llama_batch batch) {
    final p = calloc<C.llama_batch>()..ref = batch;
    return Batch(p);
  }

  /// number of tokens to alloc
  /// may be -1 if constructed from pointer directly.
  final int numTokensAlloc;

  int get numTokens => ref.n_tokens;
  set numTokens(int value) => ref.n_tokens = value;

  int get _len => numTokensAlloc > 0 ? numTokensAlloc: numTokens;

  Int32List get tokens => ref.token.asTypedList(_len);
  ffi.Pointer<C.llama_token> get tokenPtr => ref.token;

  Float32List get embd => ref.embd.asTypedList(_len);
  ffi.Pointer<ffi.Float> get embdPtr => ref.embd;

  Int32List get pos => ref.pos.asTypedList(_len);
  ffi.Pointer<C.llama_pos> get posPtr => ref.pos;

  Int32List get nSeqId => ref.n_seq_id.asTypedList(_len);
  ffi.Pointer<ffi.Int32> get nSeqIdPtr => ref.n_seq_id;

  // external ffi.Pointer<ffi.Pointer<llama_seq_id>> seq_id;
  /// 2D int32 List, (numTokens, nSeqId[i])
  List<Int32List> get seqId => List.generate(_len, (i) => ref.seq_id[i].asTypedList(nSeqId[i]));

  Int8List get logits => output;
  Int8List get output => ref.logits.asTypedList(_len);
  ffi.Pointer<ffi.Int8> get outputPtr => ref.logits;

  @override
  void dispose() {
    finalizer.detach(this);
    finalizer1.detach(this);
    C.llama_batch_free(ref);
    calloc.free(ptr);
  }

  @override
  C.llama_batch get ref => ptr.ref;
}
