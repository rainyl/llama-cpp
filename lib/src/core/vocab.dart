// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../g/llama.g.dart' as C;
import 'base.dart';
import 'enums.dart';

class Vocab extends LLAMAClass<C.llama_vocab> {
  Vocab(super.ptr);

  VocabType get type => C.llama_vocab_type(ptr);

  int get numTokens => C.llama_vocab_n_tokens(ptr);

  int get BOS => C.llama_vocab_bos(ptr);
  int get EOS => C.llama_vocab_eos(ptr);
  int get EOT => C.llama_vocab_eot(ptr);
  int get SEP => C.llama_vocab_sep(ptr);
  int get NL => C.llama_vocab_nl(ptr);
  int get PAD => C.llama_vocab_pad(ptr);
  bool get addBOS => C.llama_vocab_get_add_bos(ptr);
  bool get addEOS => C.llama_vocab_get_add_eos(ptr);
  int get FIM_PRE => C.llama_vocab_fim_pre(ptr);
  int get FIM_SUF => C.llama_vocab_fim_suf(ptr);
  int get FIM_MID => C.llama_vocab_fim_mid(ptr);
  int get FIM_PAD => C.llama_vocab_fim_pad(ptr);
  int get FIM_REP => C.llama_vocab_fim_rep(ptr);
  int get FIM_SEP => C.llama_vocab_fim_sep(ptr);

  String getText(int token) {
    final pText = C.llama_vocab_get_text(ptr, token);
    return pText.cast<Utf8>().toDartString();
  }

  double getScore(int token) => C.llama_vocab_get_score(ptr, token);

  TokenAttr getAttr(int token) => C.llama_vocab_get_attr(ptr, token);

  /// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
  bool isEOG(int token) => C.llama_vocab_is_eog(ptr, token);

  /// Identify if Token Id is a control token or a render-able token
  bool isControl(int token) => C.llama_vocab_is_control(ptr, token);

  /// Convert the provided text into tokens.
  ///
  /// [addSpecial] Allow to add BOS and EOS tokens if model is configured to do so.
  ///
  /// [parseSpecial] Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
  /// as plaintext. Does not insert a leading space.
  List<int> tokenize(
    String text, {
    bool addSpecial = true,
    bool parseSpecial = true,
    int nTokensMax = 512,
  }) {
    final ctext = text.toNativeUtf8().cast<ffi.Char>();
    ffi.Pointer<C.llama_token> pTokens = ffi.nullptr;
    int numTokens = -nTokensMax;
    try {
      while (numTokens < 0) {
        if (pTokens != ffi.nullptr) {
          calloc.free(pTokens);
        }
        pTokens = calloc<C.llama_token>(-numTokens);
        numTokens = C.llama_tokenize(
          ptr,
          ctext,
          text.nativeUtf8Length,
          pTokens,
          nTokensMax,
          addSpecial,
          parseSpecial,
        );
      }
      return pTokens.asTypedList(numTokens, finalizer: calloc.nativeFree, token: pTokens.cast());
    } finally {
      calloc.free(ctext);
    }
  }

  /// Token Id -> Piece.
  /// Uses the vocabulary in the provided context.
  /// Does not write null terminator to the buffer.
  /// User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding
  /// multiple tokens with 'add_space_prefix')
  ///
  /// [special] If true, special tokens are rendered in the output.
  ///
  /// Note: some characters may take >1 tokens, so the returned value is [Uint8List] instead of [String]
  /// to let users process it correctly.
  Uint8List tokenToPiece(
    int token, {
    int lstrip = 0,
    int maxCharNum = 128,
    bool special = true,
  }) {
    int numChars = -maxCharNum;
    ffi.Pointer<ffi.Char> textPtr = ffi.nullptr;
    while (numChars < 0) {
      if (textPtr != ffi.nullptr) {
        calloc.free(textPtr);
      }
      textPtr = calloc<ffi.Char>(-numChars);
      numChars = C.llama_token_to_piece(ptr, token, textPtr, -numChars, lstrip, special);
    }
    final codes = textPtr.cast<ffi.Uint8>().asTypedList(_length(textPtr.cast()));
    final rval = Uint8List.fromList(codes);
    calloc.free(textPtr);
    return rval;
  }

  static int _length(ffi.Pointer<ffi.Uint8> codeUnits) {
    var length = 0;
    while (codeUnits[length] != 0) {
      length++;
    }
    return length;
  }

  /// Convert the provided tokens into text (inverse of [tokenize]).
  ///
  /// [removeSpecial] Allow to remove BOS and EOS tokens if model is configured to do so.
  ///
  /// [unparseSpecial] If true, special tokens are rendered in the output.
  String detokenize(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
    int textLenMax = 1024,
  }) {
    final pTokens = calloc<C.llama_token>(tokens.length);
    pTokens.asTypedList(tokens.length).setAll(0, tokens);

    int numChars = -textLenMax;
    ffi.Pointer<ffi.Char> pText = ffi.nullptr;
    while (numChars < 0) {
      if (pText != ffi.nullptr) {
        calloc.free(pText);
      }
      pText = calloc<ffi.Char>(numChars < 0 ? -numChars : numChars);
      numChars = C.llama_detokenize(
        ptr,
        pTokens,
        tokens.length,
        pText,
        textLenMax,
        removeSpecial,
        unparseSpecial,
      );
    }

    final rval = pText.cast<Utf8>().toDartString();
    calloc.free(pTokens);
    calloc.free(pText);
    return rval;
  }

  // will be freed along with model
  @override
  void dispose() {}
}
