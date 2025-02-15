// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

import 'base.dart';
import 'enums.dart';
import '../g/llama.g.dart' as llama;

class Vocab extends LLAMAClass<llama.llama_vocab> {
  Vocab(super.ptr);

  VocabType get type => llama.llama_vocab_type(ptr);

  int get nTokens => llama.llama_vocab_n_tokens(ptr);

  int get BOS => llama.llama_vocab_bos(ptr);
  int get EOS => llama.llama_vocab_eos(ptr);
  int get EOT => llama.llama_vocab_eot(ptr);
  int get SEP => llama.llama_vocab_sep(ptr);
  int get NL => llama.llama_vocab_nl(ptr);
  int get PAD => llama.llama_vocab_pad(ptr);
  bool get addBOS => llama.llama_vocab_get_add_bos(ptr);
  bool get addEOS => llama.llama_vocab_get_add_eos(ptr);
  int get FIM_PRE => llama.llama_vocab_fim_pre(ptr);
  int get FIM_SUF => llama.llama_vocab_fim_suf(ptr);
  int get FIM_MID => llama.llama_vocab_fim_mid(ptr);
  int get FIM_PAD => llama.llama_vocab_fim_pad(ptr);
  int get FIM_REP => llama.llama_vocab_fim_rep(ptr);
  int get FIM_SEP => llama.llama_vocab_fim_sep(ptr);

  String getText(int token) {
    final pText = llama.llama_vocab_get_text(ptr, token);
    return pText.cast<Utf8>().toDartString();
  }

  double getScore(int token) => llama.llama_vocab_get_score(ptr, token);

  TokenAttr getAttr(int token) => llama.llama_vocab_get_attr(ptr, token);

  /// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
  bool isEOG(int token) => llama.llama_vocab_is_eog(ptr, token);

  /// Identify if Token Id is a control token or a render-able token
  bool isControl(int token) => llama.llama_vocab_is_control(ptr, token);

  /// @details Convert the provided text into tokens.
  /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
  /// @return Returns the number of tokens on success, no more than n_tokens_max
  /// @return Returns a negative number on failure - the number of tokens that would have been returned
  /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
  /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
  /// as plaintext. Does not insert a leading space.
  List<int> tokenize(
    String text, {
    bool addSpecial = true,
    bool parseSpecial = true,
    int nTokensMax = 512,
  }) {
    final ctext = text.toNativeUtf8().cast<ffi.Char>();
    try {
      final pTokens = calloc<llama.llama_token>(nTokensMax);
      final numTokens = llama.llama_tokenize(
        ptr,
        ctext,
        text.length,
        pTokens,
        nTokensMax,
        addSpecial,
        parseSpecial,
      );
      return pTokens.asTypedList(numTokens, finalizer: calloc.nativeFree, token: pTokens.cast());
    } finally {
      calloc.free(ctext);
    }
  }

  /// Token Id -> Piece.
  /// Uses the vocabulary in the provided context.
  /// Does not write null terminator to the buffer.
  /// User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
  /// @param special If true, special tokens are rendered in the output.
  String tokenToPiece(
    int token, {
    bool special = true,
    int lstrip = 0,
    int maxCharNum = 1024,
  }) {
    ffi.Pointer<ffi.Char> textPtr = calloc<ffi.Char>(maxCharNum);
    final numChars = llama.llama_token_to_piece(ptr, token, textPtr, maxCharNum, lstrip, special);
    if (numChars < 0) {
      calloc.free(textPtr);
      textPtr = calloc<ffi.Char>(-numChars);
      llama.llama_token_to_piece(ptr, token, textPtr, -numChars, lstrip, special);
    }
    print(numChars);
    print(textPtr.cast<ffi.Int8>().asTypedList(numChars));
    final rval = textPtr.cast<Utf8>().toDartString();
    print(rval);
    calloc.free(textPtr);
    return rval;
  }

  /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
  /// @param text The char pointer must be large enough to hold the resulting text.
  /// @return Returns the number of chars/bytes on success, no more than text_len_max.
  /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
  /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
  /// @param unparse_special If true, special tokens are rendered in the output.
  String detokenize(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
    int textLenMax = 1024,
  }) {
    final pTokens = calloc<llama.llama_token>(tokens.length);
    pTokens.asTypedList(tokens.length).setAll(0, tokens);
    int numChars = -textLenMax;
    ffi.Pointer<ffi.Char> pText = ffi.nullptr;
    while (numChars < 0) {
      if (pText != ffi.nullptr) {
        calloc.free(pText);
      }
      pText = calloc<ffi.Char>(numChars < 0 ? -numChars : numChars);
      numChars = llama.llama_detokenize(
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

  @override
  void dispose() {}
}
