import 'dart:io';

import 'package:llama_cpp/llama_cpp.dart' as llama;
import 'package:test/test.dart';

void main() {
  setUpAll(llama.backendInit);

  tearDownAll(llama.backendFree);

  const modelDir = 'test/data';

  const modelConfig = [
    ('ggml-vocab-deepseek-coder.gguf', 'ggml-vocab-deepseek-coder.gguf.inp'),
    ('ggml-vocab-deepseek-llm.gguf', 'ggml-vocab-deepseek-llm.gguf.inp'),
  ];

  for (final conf in modelConfig) {
    final modelPath = '$modelDir/${conf.$1}';
    final modelInpPath = '$modelDir/${conf.$2}';

    test('vocab.${conf.$1}', () async {
      final textList = await File(modelInpPath).readAsLines();

      final params = llama.ModelParams.create();
      expect(params.isEmpty, false);
      params.vocabOnly = true;

      final model = llama.Model.fromFile(modelPath, params: params);
      expect(model.isEmpty, false);

      final cparams = llama.ContextParams.create();
      expect(cparams.isEmpty, false);

      final ctx = llama.Context.fromModel(model, params: cparams);
      expect(ctx.isEmpty, false);

      final vocab = model.vocab;
      expect(vocab.isEmpty, false);
      expect(vocab.type, llama.VocabType.LLAMA_VOCAB_TYPE_BPE);

      final nVocab = vocab.numTokens;
      expect(nVocab, greaterThan(0));

      for (final text in textList) {
        final tokens = vocab.tokenize(text);
        final detokenized = vocab.detokenize(tokens);
        expect(detokenized, text);
      }
    });
  }
}
