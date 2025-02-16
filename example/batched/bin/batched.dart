// Batched inference example.
// An implementation of [batched](https://github.com/ggml-org/llama.cpp/tree/master/examples/batched) example of llama.cpp

import 'dart:convert';
import 'dart:math';

import 'package:args/args.dart';

import 'package:llama_cpp/llama_cpp.dart' as llama;

const String version = '0.0.1';

const defaultPrompt = "Hello my name is";

const topK = 40;
const topP = (0.9, 1);
const temperature = 0.4;
const seed = 2541;

ArgParser buildParser() {
  return ArgParser()
    ..addOption('model', abbr: 'm', mandatory: true, help: 'Path to the .gguf model file.')
    ..addOption(
      'prompt',
      abbr: 'p',
      mandatory: false,
      defaultsTo: defaultPrompt,
      help: 'The prompt to start with.',
    )
    ..addOption('n-predict', abbr: 'n', defaultsTo: '32', help: 'Number of tokens to predict.')
    ..addOption('n-parallel', defaultsTo: '8', help: 'Number of parallel threads.')
    ..addFlag('help', abbr: 'h', negatable: false, help: 'Print this usage information.')
    ..addFlag('verbose', abbr: 'v', negatable: false, help: 'Show additional command output.')
    ..addFlag('version', negatable: false, help: 'Print the tool version.');
}

void printUsage(ArgParser argParser) {
  print('Usage: dart batched.dart <flags> [arguments]');
  print(argParser.usage);
}

void commonBatchAdd(llama.Batch b, int id, int pos, List<int> seqIds, bool logits) {
  b.tokens[b.numTokens] = id;
  b.pos[b.numTokens] = pos;
  b.nSeqId[b.numTokens] = seqIds.length;
  for (var j = 0; j < seqIds.length; j++) {
    b.seqId[b.numTokens][j] = seqIds[j];
  }
  b.logits[b.numTokens] = logits ? 1 : 0;
  b.numTokens += 1;
}

void commonBatchClear(llama.Batch b) => b.numTokens = 0;

void runInference(String modelPath, String prompt, {int nPredict = 32, int nParallel = 1}) {
  llama.backendInit();
  llama.numaInit(llama.GGMLNumaStrategy.GGML_NUMA_STRATEGY_DISABLED);

  final modelParams = llama.ModelParams.create();
  final model = llama.Model.fromFile(modelPath, params: modelParams);
  if (model.isEmpty) throw Exception('Failed to load model from file: $modelPath');

  final vocab = model.vocab;
  if (vocab.isEmpty) {
    throw Exception('Failed to load vocab from model');
  }

  final tokens = vocab.tokenize(prompt);

  final nKvReq = tokens.length + (nPredict - tokens.length) * nParallel;

  final contextParams = llama.ContextParams.create();
  contextParams.nCtx = nKvReq;
  contextParams.nBatch = max(nPredict, nParallel);
  final context = llama.Context.fromModel(model, params: contextParams);
  if (context.isEmpty) {
    throw Exception('Failed to create context from model');
  }

  final samplerChainParams = llama.SamplerChainParams.create();
  final sampler = llama.Sampler.chainInit(samplerChainParams);
  if (sampler.isEmpty) {
    throw Exception('Failed to create sampler');
  }
  sampler.addSampler(llama.Sampler.topK(topK));
  sampler.addSampler(llama.Sampler.topP(topP.$1, topP.$2));
  sampler.addSampler(llama.Sampler.temp(temperature));
  sampler.addSampler(llama.Sampler.dist(seed: seed));

  final nCtx = context.nCtx;
  if (nKvReq > nCtx) {
    throw Exception('nKvReq ($nKvReq) > nCtx ($nCtx), the required KV cache size is not big enough');
  }

  for (var token in tokens) {
    print(utf8.decode(vocab.tokenToPiece(token)));
  }

  final batch = llama.Batch.init(max(tokens.length, nParallel), 0, nParallel);
  final seqIds = List.generate(nParallel, (i) => i);

  for (var (i, id) in tokens.indexed) {
    commonBatchAdd(batch, id, i, seqIds, false);
  }

  if (model.hasEncoder) {
    if (context.decode(batch) != 0) throw Exception('Failed to decode batch');
    int decoderStartTokenId = model.decoderStartToken;
    if (decoderStartTokenId == llama.LLAMA_TOKEN_NULL) {
      decoderStartTokenId = vocab.BOS;
    }
    batch.numTokens = 0;
    commonBatchAdd(batch, decoderStartTokenId, 0, seqIds, false);
  }

  // llama_decode will output logits only for the last token of the prompt
  batch.logits[batch.numTokens - 1] = 1;

  if (context.decode(batch) != 0) throw Exception('decode failed');

  // for (var i = 0; i < nParallel; i++) {
  //   context.kvCacheSeqCP(0, i, 0, batch.numTokens);
  // }

  if (nParallel > 1) print('Generating $nParallel sequences');

  // main loop

  List<String> streams = List.generate(nParallel, (i) => '');
  List<int> iBatch = List.generate(nParallel, (i) => batch.numTokens - 1);

  int nCur = batch.numTokens;
  int nDecode = 0;
  final timeMinStart = llama.timeUs();

  while (nCur <= nPredict) {
    commonBatchClear(batch);

    for (var i = 0; i < nParallel; i++) {
      if (iBatch[i] < 0) continue;

      final newTokenId = sampler.sample(context, iBatch[i]);

      if (vocab.isEOG(newTokenId) || nCur == nPredict) {
        iBatch[i] = -1;
        print('\n');
        if (nParallel > 1) print('stream $i finished at nCur=$nCur');
        continue;
      }

      if (nParallel == 1) {
        print(vocab.tokenToPiece(newTokenId));
      }

      streams[i] += utf8.decode(vocab.tokenToPiece(newTokenId));
      iBatch[i] = batch.numTokens;

      commonBatchAdd(batch, newTokenId, nCur, [i], true);

      nDecode += 1;
    }

    if (batch.numTokens == 0) break;

    nCur += 1;

    if (context.decode(batch) != 0) {
      throw Exception('failed to eval');
    }
  }

  if (nParallel > 1) {
    for (var i = 0; i < nParallel; i++) {
      print("sequence $i: \n\n $prompt${streams[i]}\n\n");
    }
  }

  final timeMainEnd = llama.timeUs();
  print(
    'Decoded $nDecode tokens in ${(timeMainEnd - timeMinStart) / 1000000.0} s, '
    'speed: ${nDecode / ((timeMainEnd - timeMinStart) / 1000000.0)}',
  );

  batch.dispose();
  sampler.dispose();
  context.dispose();
  model.dispose();
  llama.backendFree();
}

void main(List<String> arguments) {
  final ArgParser argParser = buildParser();
  try {
    final ArgResults results = argParser.parse(arguments);
    bool verbose = false;

    // Process the parsed arguments.
    if (results.flag('help')) {
      printUsage(argParser);
      return;
    }
    if (results.flag('version')) {
      print('batched version: $version');
      return;
    }
    if (results.flag('verbose')) {
      verbose = true;
    }

    final modelPath = results['model'] as String;
    final prompt = results['prompt'] as String;
    final nPredict = int.parse(results['n-predict'] as String, radix: 10);
    final nParallel = int.parse(results['n-parallel'] as String, radix: 10);
    runInference(modelPath, prompt, nPredict: nPredict, nParallel: nParallel);

    // Act on the arguments provided.
    print('Positional arguments: ${results.rest}');
    if (verbose) {
      print('[VERBOSE] All arguments: ${results.arguments}');
    }
  } on FormatException catch (e) {
    // Print usage information if an invalid argument was provided.
    print(e.message);
    print('');
    printUsage(argParser);
  }
}
