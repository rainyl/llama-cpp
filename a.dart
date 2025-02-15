import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'package:llama_cpp/llama_cpp.dart' as llama;

void main(List<String> args) {
  final modelPath = "/Users/rainy/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf";
  final model = llama.Model.fromFile(modelPath);
  final vocab = model.vocab;
  final tokens = vocab.tokenize("Hello World, 你好 世界 🌹 1234567890!@#\$%^&*()_+-=[]\\{}|;':\",./<>?`~ ");
  print(tokens);
  final str = vocab.detokenize(tokens);
  print(str);
  final pieces = tokens.map((e) => vocab.tokenToPiece(e)).toList();
  print(pieces);
}
