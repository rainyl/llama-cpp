import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:llama_cpp/src/core/model_params.dart';

import 'context.dart';
import 'token_data.dart';
import 'base.dart';
import '../g/llama.g.dart' as llama;
import 'vocab.dart';

class Sampler extends LLAMAStruct<llama.llama_sampler> {
  static final finalizer = ffi.NativeFinalizer(llama.addresses.llama_sampler_free.cast());

  Sampler(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  // external ffi.Pointer<llama_sampler> llama_sampler_init(
  //   ffi.Pointer<llama_sampler_i> iface,
  //   llama_sampler_context_t ctx,
  // );
  // factory Sampler.init(){}

  factory Sampler.greedy() {
    final p = llama.llama_sampler_init_greedy();
    return Sampler(p);
  }

  factory Sampler.dist({int seed = 0}) {
    final p = llama.llama_sampler_init_dist(seed);
    return Sampler(p);
  }

  /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  factory Sampler.topK(int k) {
    final p = llama.llama_sampler_init_top_k(k);
    return Sampler(p);
  }

  /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  factory Sampler.topP(double p, int minKeep) {
    final ptr = llama.llama_sampler_init_top_p(p, minKeep);
    return Sampler(ptr);
  }

  /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
  factory Sampler.minP(double p, int minKeep) {
    final ptr = llama.llama_sampler_init_min_p(p, minKeep);
    return Sampler(ptr);
  }

  /// Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
  factory Sampler.typical(double p, int minKeep) {
    final ptr = llama.llama_sampler_init_typical(p, minKeep);
    return Sampler(ptr);
  }

  /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  factory Sampler.temp(double t) {
    final ptr = llama.llama_sampler_init_temp(t);
    return Sampler(ptr);
  }

  /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
  factory Sampler.tempExt(
    double t,
    double delta,
    double exponent,
  ) {
    final ptr = llama.llama_sampler_init_temp_ext(t, delta, exponent);
    return Sampler(ptr);
  }

  /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
  factory Sampler.xtc({
    double p = 0.0,
    double t = 0.0,
    int minKeep = 0,
    int seed = 0,
  }) {
    final ptr = llama.llama_sampler_init_xtc(p, t, minKeep, seed);
    return Sampler(ptr);
  }

  /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
  factory Sampler.topNSigma(double n) {
    final ptr = llama.llama_sampler_init_top_n_sigma(n);
    return Sampler(ptr);
  }

  /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  factory Sampler.mirostat(int nVocab, int seed, double tau, double eta, int m) {
    final ptr = llama.llama_sampler_init_mirostat(nVocab, seed, tau, eta, m);
    return Sampler(ptr);
  }

  /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  factory Sampler.mirostatV2(int seed, double tau, double eta) {
    final ptr = llama.llama_sampler_init_mirostat_v2(seed, tau, eta);
    return Sampler(ptr);
  }

  factory Sampler.grammar(Vocab vocab, String grammarStr, String grammarRoot) {
    final cgs = grammarStr.toNativeUtf8().cast<ffi.Char>();
    final cgr = grammarRoot.toNativeUtf8().cast<ffi.Char>();
    final ptr = llama.llama_sampler_init_grammar(vocab.ptr, cgs, cgr);
    calloc.free(cgs);
    calloc.free(cgr);
    return Sampler(ptr);
  }

  /// @details Lazy grammar sampler, introduced in https://github.com/ggerganov/llama.cpp/pull/9639
  /// @param trigger_words A list of words that will trigger the grammar sampler. This may be updated to a loose regex syntax (w/ ^) in a near future.
  /// @param trigger_tokens A list of tokens that will trigger the grammar sampler.
  factory Sampler.grammarLazy(
    Vocab vocab,
    String grammarStr,
    String grammarRoot,
    List<String> triggerWords,
    List<int> triggerTokens,
  ) {
    final cgs = grammarStr.toNativeUtf8().cast<ffi.Char>();
    final cgr = grammarRoot.toNativeUtf8().cast<ffi.Char>();
    final ctw = calloc<ffi.Pointer<ffi.Char>>(triggerWords.length);
    for (var i = 0; i < triggerWords.length; i++) {
      final ct = triggerWords[i].toNativeUtf8().cast<ffi.Char>();
      ctw[i] = ct;
    }
    final ctt = calloc<llama.llama_token>(triggerTokens.length);
    for (var i = 0; i < triggerTokens.length; i++) {
      final ct = triggerTokens[i];
      ctt[i] = ct;
    }
    final ptr = llama.llama_sampler_init_grammar_lazy(
      vocab.ptr,
      cgs,
      cgr,
      ctw,
      triggerWords.length,
      ctt,
      triggerTokens.length,
    );
    calloc.free(cgs);
    calloc.free(cgr);
    for (var i = 0; i < triggerWords.length; i++) {
      calloc.free(ctw[i]);
    }
    calloc.free(ctt);
    return Sampler(ptr);
  }

  /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow.
  /// For example, apply top-k or top-p sampling first.
  factory Sampler.penalties(
    int penaltyLastN,
    double penaltyRepeat,
    double penaltyFreq,
    double penaltyPresent,
  ) {
    final ptr = llama.llama_sampler_init_penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent);
    return Sampler(ptr);
  }

  /// @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
  factory Sampler.dry(
    Vocab vocab,
    int nCtxTrain,
    double dryMultiplier,
    double dryBase,
    int dryAllowedLength,
    int dryPenaltyLastN,
    List<String> seqBreakers,
  ) {
    final csb = calloc<ffi.Pointer<ffi.Char>>(seqBreakers.length);
    for (var i = 0; i < seqBreakers.length; i++) {
      final ct = seqBreakers[i].toNativeUtf8().cast<ffi.Char>();
      csb[i] = ct;
    }
    final ptr = llama.llama_sampler_init_dry(
      vocab.ptr,
      nCtxTrain,
      dryMultiplier,
      dryBase,
      dryAllowedLength,
      dryPenaltyLastN,
      csb,
      seqBreakers.length,
    );
    for (var i = 0; i < seqBreakers.length; i++) {
      calloc.free(csb[i]);
    }
    calloc.free(csb); // TODO: necessary?
    return Sampler(ptr);
  }

  factory Sampler.logitBias(
    int nVocab,
    List<LogitBias> logitBias,
  ) {
    final ctb = calloc<llama.llama_logit_bias>(logitBias.length);
    for (var i = 0; i < logitBias.length; i++) {
      final ct = logitBias[i];
      ctb[i] = ct.ref;
    }
    final ptr = llama.llama_sampler_init_logit_bias(nVocab, logitBias.length, ctb);
    calloc.free(ctb);
    return Sampler(ptr);
  }

  /// this sampler is meant to be used for fill-in-the-middle infilling
  /// it's supposed to be used after top_k + top_p sampling
  ///
  /// 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
  /// 2. combine probs of tokens that have the same prefix
  ///
  /// example:
  ///
  /// - before:
  /// "hel":   0.5
  /// "hell":  0.2
  /// "hello": 0.1
  /// "dummy": 0.1
  ///
  /// - after:
  /// "hel":   0.8
  /// "dummy": 0.1
  ///
  /// 3. discard non-EOG tokens with low prob
  /// 4. if no tokens are left -> pick EOT
  factory Sampler.infill(Vocab vocab) {
    final ptr = llama.llama_sampler_init_infill(vocab.ptr);
    return Sampler(ptr);
  }

  // ffi.Pointer<llama_sampler_i> iface;
  // ISampler? _iface;

  llama.llama_sampler_context_t get ctx => ptr.ref.ctx;
  set ctx(llama.llama_sampler_context_t value) => ptr.ref.ctx = value;

  int get seed => llama.llama_sampler_get_seed(ptr);

  String get name => llama.llama_sampler_name(ptr).cast<Utf8>().toDartString();

  /// @details Sample and accept a token from the idx-th output of the last evaluation
  ///
  /// Shorthand for:
  /// const auto * logits = llama_get_logits_ith(ctx, idx);
  /// llama_token_data_array cur_p = { ... init from logits ... };
  /// llama_sampler_apply(smpl, &cur_p);
  /// auto token = cur_p.data[cur_p.selected].id;
  /// llama_sampler_accept(smpl, token);
  /// return token;
  /// Returns the sampled token
  int sample(Context ctx, int idx) => llama.llama_sampler_sample(ptr, ctx.ptr, idx);

  void accept(int token) => llama.llama_sampler_accept(ptr, token);

  void apply(TokenDataArray curP) => llama.llama_sampler_apply(ptr, curP.ptr);

  void reset() => llama.llama_sampler_reset(ptr);

  Sampler clone() => Sampler(llama.llama_sampler_clone(ptr));

  /// llama_sampler_chain
  /// a type of llama_sampler that can chain multiple samplers one after another
  factory Sampler.chainInit(SamplerChainParams params) {
    final ptr = llama.llama_sampler_chain_init(params.ref);
    return Sampler(ptr);
  }

  /// important: takes ownership of the sampler object and will free it when llama_sampler_free is called
  void addSampler(Sampler smpl) => llama.llama_sampler_chain_add(ptr, smpl.ptr);

  Sampler getSampler(int i) => Sampler(llama.llama_sampler_chain_get(ptr, i));

  /// after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
  void removeSampler(int i) => llama.llama_sampler_chain_remove(ptr, i);

  int getChainLength() => llama.llama_sampler_chain_n(ptr);

  @override
  void dispose() {
    finalizer.detach(this);
    llama.llama_sampler_free(ptr);
  }

  @override
  llama.llama_sampler get ref => ptr.ref;
}

// typedef ISamplerApplyFunction = ffi.Void Function(
//     ffi.Pointer<llama.llama_sampler> smpl, ffi.Pointer<llama.llama_token_data_array> curP);

// abstract class ISampler extends LLAMAStruct<llama.llama_sampler_i> {
//   ISampler(super.ptr);

//   factory ISampler.create({required void Function(Sampler smpl, TokenDataArray curP) onApply}) {
//     final p = calloc<llama.llama_sampler_i>();

//     late final ffi.NativeCallable<ISamplerApplyFunction> callback;

//     void _onApply(ffi.Pointer<llama.llama_sampler> smpl, ffi.Pointer<llama.llama_token_data_array> curP) {
//       onApply(Sampler(smpl), TokenDataArray(curP));
//       callback.close();
//     }

//     callback = ffi.NativeCallable.listener(_onApply);
//     return ISampler(p);
//   }

//   @override
//   void dispose() {
//     // TODO: implement dispose
//   }

//   @override
//   // TODO: implement ref
//   llama.llama_sampler_i get ref => throw UnimplementedError();

//   // /// can be NULL
//   // String Function(Sampler smpl)? name;

//   // /// can be NULL
//   // void Function(Sampler smpl, int token)? accept;

//   // /// required
//   // void Function(Sampler smpl, TokenDataArray curP)? apply;

//   // /// can be NULL
//   // void Function(Sampler smpl)? reset;

//   // /// can be NULL if ctx is NULL
//   // Sampler Function(Sampler smpl)? clone;

//   // /// can be NULL if ctx is NULL
//   // void Function(Sampler smpl)? free;
// }

class LogitBias extends LLAMAStruct<llama.llama_logit_bias> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  LogitBias(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory LogitBias.create(int token, double bias) {
    final p = calloc<llama.llama_logit_bias>()
      ..ref.token = token
      ..ref.bias = bias;
    return LogitBias(p);
  }

  int get token => ref.token;
  set token(int value) => ref.token = value;

  double get bias => ref.bias;
  set bias(double value) => ref.bias = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  llama.llama_logit_bias get ref => ptr.ref;
}
