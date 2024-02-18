import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'dart:io' show Platform, stderr;
import 'dart:math' show max;

import 'ffi.dart';
import '../native_llama_cpp.dart' as llama_cpp;
import 'sampling.dart';

/// Params holder like `gpt_params` in `common/common.h`
final class LlamaParams {
  final int seed;
  final int nThread;
  final int nThreadBatch;
  final int nPredict;
  final int nCtx;
  final int nBatch;
  final int? nGpuLayers;
  final int? mainGpu;
  final bool numa;

  const LlamaParams(
    this.seed,
    this.nThread,
    this.nThreadBatch,
    this.nPredict,
    this.nCtx,
    this.nBatch,
    this.nGpuLayers,
    this.mainGpu,
    this.numa,
  );
}

String _systemInfo(LlamaParams params) {
  final batch = params.nThreadBatch != -1
      ? ' (n_threads_batch = ${params.nThreadBatch})'
      : '';
  return 'system_info: n_threads = ${params.nThread}$batch '
      '/ ${Platform.numberOfProcessors} '
      '| ${NativeString.fromNative(llama_cpp.llama_print_system_info())}';
}

bool _shouldAddBosToken(ffi.Pointer<llama_cpp.llama_model> model) {
  final addBos = llama_cpp.llama_add_bos_token(model);
  return addBos != -1 ? addBos != 0
      : llama_cpp.llama_vocab_type1(model) == 0; // LLAMA_VOCAB_TYPE_SPM
}

/// A class represent native llama data structure, run in separate isolate.
final class NativeLLama {
  static const engTag = '__end__';
  static const closeTag = '__close__';

  final ffi.Pointer<llama_cpp.llama_model> model;
  final ffi.Pointer<llama_cpp.llama_context> ctx;
  final llama_cpp.llama_batch batch;
  final NativeString cStr;
  final bool verbose;
  final tokenBuf = TokenArray(size: 64);
  final array = TokenDataArray(512);

  NativeLLama._(
    this.model,
    this.ctx,
    this.batch,
    this.cStr,
    this.verbose,
  );

  factory NativeLLama(String path, LlamaParams params, {
    bool verbose = false,
  }) {
    final ctxSize = max(params.nCtx, 8);
    final seed = params.seed > 0 ? params.seed
        : DateTime.now().millisecondsSinceEpoch ~/ 1000;
    print('seed = $seed');
    print('llama backend init');
    llama_cpp.llama_backend_init(params.numa);

    final cStr = NativeString();
    final modelParams = llama_cpp.llama_model_default_params();
    final nGpuLayers = params.nGpuLayers;
    if (nGpuLayers != null) {
      modelParams.n_gpu_layers = nGpuLayers > 0 ? nGpuLayers : 0;
    }
    final mainGpu = params.mainGpu;
    if (mainGpu != null) {
      modelParams.main_gpu = mainGpu;
    }

    final model =
        llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);
    if (model.address == 0) {
      throw Exception("Load model from '$path' failed");
    }

    final t = params.nThread;
    final ctxParams = llama_cpp.llama_context_default_params()
      ..n_ctx = ctxSize
      ..n_batch = params.nBatch
      ..seed = seed
      ..n_threads = t
      ..n_threads_batch = params.nThreadBatch == -1 ? t : params.nThreadBatch;
    final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
    if (ctx.address == 0) {
      throw Exception("Create llama context failed");
    }

    final nCtxTrain = llama_cpp.llama_n_ctx_train(model);
    final nCtx = llama_cpp.llama_n_ctx(ctx);
    print('n_ctx: $nCtx, train=$nCtxTrain');
    if (nCtx > nCtxTrain) {
      print('warning: model was trained on only $nCtxTrain context tokens '
          '($nCtx specified)');
    }
    print(_systemInfo(params));
    print('add_bos: ${_shouldAddBosToken(model)}');

    final batch = llama_cpp.llama_batch_init(ctxParams.n_batch, 0, 1);

    return NativeLLama._(
      model,
      ctx,
      batch,
      cStr,
      verbose,
    );
  }

  /// free native resource, need explicitly calling.
  void dispose() {
    array.dispose();
    tokenBuf.dispose();
    cStr.dispose();

    llama_cpp.llama_batch_free(batch);
    llama_cpp.llama_free(ctx);
    llama_cpp.llama_free_model(model);
    llama_cpp.llama_backend_free();
  }

  void _log(String str, {bool console = true}) {
    final leadingNewLine = str.startsWith('\n');
    stderr.writeln(leadingNewLine ? str.replaceFirst('\n', '\n  ') : '  $str');
  }

  /// Generate token string by @prompt.
  /// Some model would return two tokens to represent a single word,
  /// so it is better to use raw stream.
  Stream<List<int>> generate(String prompt, {
    int? nPrev,
    int? nProbs,
    int? topK,
    double? topP,
    double? minP,
    double? tfsZ,
    double? typicalP,
    double? temperature,
    int? penaltyLastN,
    double? penaltyRepeat,
    double? penaltyFrequency,
    double? penaltyPresent,
    int? mirostat,
    double? mirostatTau,
    double? mirostatEta,
    bool? penalizeNewline,
    String? samplersSequence,
  }) async* {
    prompt.into(cStr);
    tokenBuf.pavedBy(model, cStr);
    if (verbose) {
      _log('prompt: "$prompt"');
      _log('tokens: ${_tokensString(tokenBuf.pointerAt(0), tokenBuf.length)}');
    }
    final eosToken = llama_cpp.llama_token_eos(model);

    var num = 0;
    var code = 0;

    final defaultParams = const SamplingParams();
    final params = SamplingParams(
      nPrev: nPrev ?? defaultParams.nPrev,
      nProbs: nProbs ?? defaultParams.nProbs,
      topK: topK ?? defaultParams.topK,
      topP: topP ?? defaultParams.topP,
      minP: minP ?? defaultParams.minP,
      tfsZ: tfsZ ?? defaultParams.tfsZ,
      typicalP: typicalP ?? defaultParams.typicalP,
      temperature: temperature ?? defaultParams.temperature,
      penaltyLastN: penaltyLastN ?? defaultParams.penaltyLastN,
      penaltyRepeat: penaltyRepeat ?? defaultParams.penaltyRepeat,
      penaltyFrequency: penaltyFrequency ?? defaultParams.penaltyFrequency,
      penaltyPresent: penaltyPresent ?? defaultParams.penaltyPresent,
      mirostat: mirostat ?? defaultParams.mirostat,
      mirostatTau: mirostatTau ?? defaultParams.mirostatTau,
      mirostatEta: mirostatEta ?? defaultParams.mirostatEta,
      penalizeNewline: penalizeNewline ?? defaultParams.penalizeNewline,
      samplersSequence: samplersSequence ?? defaultParams.samplersSequence,
    );
    if (verbose) {
      _log('sampling:\n$params');
      _log('sampling order:\n${params.samplingOrder}');
      _log('generate: n_ctx = ${llama_cpp.llama_n_ctx(ctx)}, '
          'n_batch = ${llama_cpp.llama_n_batch(ctx)}, '
          'n_predict = %d, '
          'n_keep = %d');
    }
    final ctxSampling = SamplingContext.from(params);
    ctxSampling.acceptSampling(
      ctx,
      tokenBuf.toList(),
      false,
    );
    llama_cpp.llama_reset_timings(ctx);
    llama_cpp.llama_kv_cache_clear(ctx);
    while ((code = _decodeBatch(num, num == 0)) == 0) {
      if (verbose) {
        _log('<<<<<<<<<');
        _log('eval: ${_tokensString(tokenBuf.pointerAt(0), tokenBuf.length)}',
            console: false);
      }
      final tokenId = _sampleSampling(ctxSampling, batch.n_tokens - 1);
      if (verbose) {
        _log('sampled token(${params.mirostat}): ${"$tokenId".padLeft(8)}: ',
            console: false);
      }
      if (tokenId == eosToken) {
        code = 3;
        break;
      }
      final token = cStr.tokenBytes(model, tokenId);
      yield token;
      ctxSampling.acceptSampling(ctx, [tokenId], true);
      if (verbose) {
        _log('\nlast: ${_tokensString(ctxSampling.penaltyPointer,
            ctxSampling.usedSize)}', console: false);
        _log('>>>>>>>>>');
      }

      num += batch.n_tokens;
      batch.n_tokens = 0;
      tokenBuf
        ..clear()
        ..add(tokenId);
    }
    llama_cpp.llama_print_timings(ctx);
    if (verbose) {
      _log("sample llama logits finished with '$code'.");
    }
    ctxSampling.free();
    yield utf8.encode(engTag);
  }

  int _decodeBatch(int count, bool init) {
    final tokenNum = tokenBuf.length;
    // evaluate the initial prompt
    for (var i = 0; i < tokenNum; i++) {
      _addLlamaBatch(tokenBuf[i], count + i, !init);
    }
    if (init) {
      batch.logits[batch.n_tokens - 1] = 1;
    }
    return llama_cpp.llama_decode(ctx, batch);
  }

  void _addLlamaBatch(int id, int pos, bool logits) {
    final n = batch.n_tokens;
    batch.token[n] = id;
    batch.pos[n] = pos;
    batch.n_seq_id[n] = 1;
    batch.seq_id[n][0] = 0;
    batch.logits[n] = logits ? 1 : 0;

    batch.n_tokens++;
  }

  int _sampleSampling(SamplingContext ctxSampling, int idx,
      [bool isResampling = false]) {
    final params = ctxSampling.params;
    final model = llama_cpp.llama_get_model(ctx);
    final nVocab = llama_cpp.llama_n_vocab(model);
    final temp = params.temperature;
    final penaltyRepeat = params.penaltyRepeat;
    final penaltyFrequency = params.penaltyFrequency;
    final penaltyPresent = params.penaltyPresent;
    final mirostat = params.mirostat;
    final mirostatTau = params.mirostatTau;
    final mirostatEta = params.mirostatEta;
    final penalizeNewline = params.penalizeNewline;

    final logits = llama_cpp.llama_get_logits_ith(ctx, idx);
    final logitBias = params.logitBias?.entries;
    logitBias?.forEach((e) {
      logits[e.key] += e.value;
    });
    array.pavedBy(logits, nVocab);
    // apply penalties
    final usedSize = ctxSampling.usedSize;
    if (usedSize > 0) {
      final nl = llama_cpp.llama_token_nl(model);
      final logit = logits[nl];
      llama_cpp.llama_sample_repetition_penalties(
        ctx,
        array.pointer,
        ctxSampling.penaltyPointer,
        usedSize,
        penaltyRepeat,
        penaltyFrequency,
        penaltyPresent,
      );
      if (!penalizeNewline) {
        for (var i = 0; i < array.length; i++) {
          final data = array[i];
          if (data.id == nl) {
            final old = data.logit;
            array.setLogit(i, logit);
            final v = array[i].logit;
            _log("$i: $old -> $v", console: false);
            break;
          }
        }
      }
    }

    final grammar = ctxSampling.grammar;
    if (isResampling && grammar != null) {
      llama_cpp.llama_sample_grammar(ctx, array.pointer, grammar);
    }
    var id = 0;
    if (temp < 0.0) {
      llama_cpp.llama_sample_softmax(ctx, array.pointer);
      id = array[0].id;
    } else if (temp == 0.0) {
      id = llama_cpp.llama_sample_token_greedy(ctx, array.pointer);
    } else {
      if (mirostat == 1) {
        const mirostatM = 100;
        llama_cpp.llama_sample_temp(ctx, array.pointer, temp);
        id = llama_cpp.llama_sample_token_mirostat(ctx, array.pointer,
            mirostatTau, mirostatEta, mirostatM, ctxSampling.mirostatMu);
      } else if (mirostat == 2) {
        llama_cpp.llama_sample_temp(ctx, array.pointer, temp);
        id = llama_cpp.llama_sample_token_mirostat_v2(ctx, array.pointer,
            mirostatTau, mirostatEta, ctxSampling.mirostatMu);
      } else {
        final minKeep = max(1, params.nProbs);
        _samplerQueue(params, nVocab, minKeep);
        id = llama_cpp.llama_sample_token(ctx, array.pointer);
      }
    }

    if (grammar != null && !isResampling) {
      // TODO: consider grammar
    }

    return id;
  }

  void _samplerQueue(SamplingParams params, int capacity, int minKeep) {
    final topK = params.topK <= 0 ? capacity : params.topK;
    for (final i in params.samplersSequence.codeUnits) {
      switch (i) {
        case kChar:
          llama_cpp.llama_sample_top_k(ctx, array.pointer, topK, minKeep);
          break;
        case fChar:
          llama_cpp.llama_sample_tail_free(ctx, array.pointer,
              params.tfsZ, minKeep);
          break;
        case yChar:
          llama_cpp.llama_sample_typical(ctx, array.pointer,
              params.typicalP, minKeep);
          break;
        case pChar:
          llama_cpp.llama_sample_top_p(ctx, array.pointer,
              params.topP, minKeep);
          break;
        case mChar:
          llama_cpp.llama_sample_min_p(ctx, array.pointer,
              params.minP, minKeep);
          break;
        case tChar:
          llama_cpp.llama_sample_temp(ctx, array.pointer, params.temperature);
          break;
        default:
          break;
      }
    }
  }

  String _tokensString(ffi.Pointer<llama_cpp.llama_token> pointer, int len) {
    final buf = StringBuffer('[');
    for (var i = 0; i < len; i++) {
      final id = pointer[i];
      buf.write("'${cStr.tokenString(model, id)}':$id, ");
    }
    buf.write(']');
    return buf.toString();
  }
}
