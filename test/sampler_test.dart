import 'dart:math' as math;

import 'package:llama_cpp/llama_cpp.dart' as llama;
import 'package:test/test.dart';

import 'matcher_iterable_num.dart';

bool iterCloseTo(Iterable<num> a, Iterable<num> b, {double epsilon = 1e-3}) {
  return a.length == b.length && a.indexed.every((e) => e.$2 - b.elementAt(e.$1) < epsilon);
}

const seed = 0;

void main() {
  group('Sampler', () {
    test('.temp', () {
      void testFunc(List<double> probs, List<double> probsExpected, double temp) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.temp(temp)..apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 1.0);
      testFunc([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0], 0.0);
    });

    test('.tempExt', () {
      void testFunc(List<double> probs, List<double> probsExpected, double temp, double delta, double exp) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.tempExt(temp, delta, exp)..apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 1.0, 0.0, 1.0);
      testFunc([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 1.0);
    });

    test('.topK', () {
      void testFunc(List<double> probs, List<double> probsExpected, int k) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.topK(k)..apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.1, 0.2, 0.3, 0.4], [1.0], 1);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.44444, 0.33333, 0.22222], 3);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 4);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 0);
    });

    test('.topP', () {
      void testFunc(List<double> probs, List<double> probsExpected, double p, {int minKeep = 1}) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.topP(p, minKeep)..apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.1, 0.2, 0.3, 0.4], [1.0], 0.0);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.571429, 0.428571], 0.7);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.44444, 0.33333, 0.22222], 0.8);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 1.0);
    });

    test('.minP', () {
      void testFunc(List<double> probs, List<double> probsExpected, double p, {int minKeep = 1}) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.minP(p, minKeep)..apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 1.0, 0.3 / 1.0, 0.2 / 1.0, 0.1 / 1.0], 0.0);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 1.0, 0.3 / 1.0, 0.2 / 1.0, 0.1 / 1.0], 0.24);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9], 0.26);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9], 0.49);

      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.7, 0.3 / 0.7], 0.51);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.7, 0.3 / 0.7], 0.74);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.4], 0.76);
      testFunc([0.1, 0.2, 0.3, 0.4], [0.4 / 0.4], 1.00);
    });

    test('.xtc', () {
      void testFunc(List<double> probs, List<double> probsExpected, double p, double t) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.xtc(p: p, t: t)..apply(curP);
        sampler.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.4, 0.3, 0.2, 0.1], [0.1], 0.99, 0.09);
      testFunc([0.4, 0.3, 0.2, 0.1], [0.2, 0.1], 0.99, 0.19);
      testFunc([0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1], 0.99, 0.29);
      // XTC should not:
      // testFunc([0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], 0.99, 0.39);
    });

    test('.typical', () {
      void testFunc(List<double> probs, List<double> probsExpected, double p) {
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler = llama.Sampler.typical(p, 1)..apply(curP);
        sampler.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.97, 0.01, 0.01, 0.01], [0.97], 0.5);
      testFunc([0.4, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 0.5);
    });

    test('.penalties', () {
      void testFunc(
        List<double> probs,
        List<int> lastTokens,
        List<double> probsExpected,
        double repeatPenalty,
        double alphaFrequency,
        double alphaPresence,
      ) {
        expect(probs.length, probsExpected.length);
        final curP = llama.TokenDataArray.generate(probs.length, (i) {
          final logit = math.log(probs[i]);
          return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
        });
        final sampler =
            llama.Sampler.penalties(lastTokens.length, repeatPenalty, alphaFrequency, alphaPresence);
        lastTokens.forEach(sampler.accept);
        sampler.apply(curP);
        sampler.dispose();

        final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
        sampler1.dispose();

        expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
      }

      testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0.25, 0.25, 0.25, 0.25, 0], 50, 0, 0);
      testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2], [0.5, 0.5, 0, 0, 0], 50, 0, 0);
      testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2, 0, 0], [0.5, 0.5, 0, 0, 0], 50, 0, 0);

      testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0.249997, 0.249997, 0.249997, 0.249997, 0.000011], 1, 5, 5);
      testFunc(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0, 1, 2],
        [0.499966, 0.499966, 0.000023, 0.000023, 0.000023],
        1,
        5,
        5,
      );
      testFunc(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0, 1, 2, 0, 0],
        [0.499977, 0.499977, 0.000023, 0.000023, 0.000000],
        1,
        5,
        5,
      );
    });

    // test('.dry', () {
    //   void testFunc(
    //     List<double> probs,
    //     List<int> lastTokens,
    //     List<double> probsExpected,
    //     double dryMultiplier,
    //     double dryBase,
    //     int dryAllowedLength,
    //     int dryPenaltyLastN,
    //     List<List<int>> seqBreakers,
    //   ) {
    //     expect(probs.length, probsExpected.length);
    //     final curP = llama.TokenDataArray.generate(probs.length, (i) {
    //       final logit = math.log(probs[i]);
    //       return llama.TokenData.create(id: i, logit: logit, p: probs[i]);
    //     });

    //     final vocab = llama.Vocab(ptr);
    //     final sampler =
    //         llama.Sampler.dry(lastTokens.length, repeatPenalty, alphaFrequency, alphaPresence);
    //     lastTokens.forEach(sampler.accept);
    //     sampler.apply(curP);
    //     sampler.dispose();

    //     final sampler1 = llama.Sampler.dist(seed: seed)..apply(curP);
    //     sampler1.dispose();

    //     expect(curP.toList().map((e) => e.p), iterableCloseTo(probsExpected, 1e-3));
    //   }

    //   testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0.25, 0.25, 0.25, 0.25, 0], 50, 0, 0);
    //   testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2], [0.5, 0.5, 0, 0, 0], 50, 0, 0);
    //   testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2, 0, 0], [0.5, 0.5, 0, 0, 0], 50, 0, 0);

    //   testFunc([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0.249997, 0.249997, 0.249997, 0.249997, 0.000011], 1, 5, 5);
    //   testFunc(
    //     [0.2, 0.2, 0.2, 0.2, 0.2],
    //     [0, 1, 2],
    //     [0.499966, 0.499966, 0.000023, 0.000023, 0.000023],
    //     1,
    //     5,
    //     5,
    //   );
    //   testFunc(
    //     [0.2, 0.2, 0.2, 0.2, 0.2],
    //     [0, 1, 2, 0, 0],
    //     [0.499977, 0.499977, 0.000023, 0.000023, 0.000000],
    //     1,
    //     5,
    //     5,
    //   );
    // });

    test('.samplerQueue', () {
      void testFunc(int nVocab, String samplersSequence, int topK, double topP, double minP) {
        final curP = llama.TokenDataArray.generate(nVocab, (i) {
          final logit = math.log(i);
          return llama.TokenData.create(id: i, logit: logit, p: 0.0);
        });

        var minTokenId = 0;
        var maxTokenId = nVocab - 1;
        for (final s in samplersSequence.split('')) {
          switch (s) {
            case 'k':
              llama.Sampler.topK(topK)
                ..apply(curP)
                ..dispose();
            case 'y':
              throw UnimplementedError('Invalid sampler');
            case 'p':
              llama.Sampler.topP(topP, 1)
                ..apply(curP)
                ..dispose();
            case 'm':
              llama.Sampler.minP(minP, 1)
                ..apply(curP)
                ..dispose();
            case 't':
              throw UnimplementedError('Invalid sampler');
            default:
              throw UnimplementedError('Invalid sampler');
          }

          llama.Sampler.dist(seed: seed)
            ..apply(curP)
            ..dispose();

          final size = curP.size;
          if (s == 'k') {
            final expectedSize = math.min(size, topK);
            minTokenId = math.max(minTokenId, nVocab - topK);
            expect(size, expectedSize);
            expect(curP.toList()[0].id, maxTokenId);
            expect(curP.toList()[expectedSize - 1].id, minTokenId);
          } else if (s == 'p') {
            final int softmaxDivisor =
                (nVocab * (nVocab - 1) / 2 - minTokenId * (minTokenId - 1) / 2).toInt();
            final softmaxNumeratorTarget = (topP * softmaxDivisor).ceil();
            minTokenId = nVocab;
            int expectedSize = 0;
            int cumsum = 0;
            do {
              minTokenId--;
              expectedSize++;
              cumsum += minTokenId;
            } while (cumsum < softmaxNumeratorTarget);

            if (minTokenId == 1) {
              minTokenId--;
              expectedSize += 1;
            }

            expect(size, expectedSize);
            expect(curP.toList()[0].id, maxTokenId);
            expect(curP.toList()[expectedSize - 1].id, minTokenId);
          } else if (s == "m") {
            int expectedSize = ((1.0 - minP) * nVocab).ceil();
            expectedSize = math.max(expectedSize, 1);
            expectedSize = math.min(expectedSize, size);

            minTokenId = (minP * nVocab).floor();
            minTokenId = math.max(minTokenId, 1);
            minTokenId = math.max(minTokenId, nVocab - size);
            minTokenId = math.min(minTokenId, nVocab - 1);

            expect(size, expectedSize);
            expect(curP.toList()[0].id, maxTokenId);
            expect(curP.toList()[expectedSize - 1].id, minTokenId);
          } else {
            throw UnimplementedError('Invalid sampler');
          }
        }
      }

      testFunc(10000, "k", 10000, 1.0, 1.0);
      testFunc(10000, "k", 1, 1.0, 1.0);
      testFunc(10000, "p", 10000, 1.0, 1.0);
      testFunc(10000, "p", 10000, 0.0, 1.0);
      testFunc(10000, "m", 10000, 1.0, 1.0);
      testFunc(10000, "m", 10000, 1.0, 1e-12);

      testFunc(10000, "k", 100, 1.0000, 1.0);
      testFunc(10000, "p", 10000, 0.0002, 1.0);
      testFunc(10000, "p", 10000, 0.8000, 1.0);
      testFunc(10000, "m", 10000, 1.0000, 9997.9 / 9999.0);
      testFunc(10000, "m", 10000, 1.0000, 0.1);

      testFunc(10000, "kp", 100, 0.8, 0.1);
      testFunc(10000, "km", 100, 0.8, 0.1);
      testFunc(10000, "pk", 100, 0.8, 0.1);
      testFunc(10000, "pm", 100, 0.8, 0.1);
      testFunc(10000, "mk", 100, 0.8, 0.1);
      testFunc(10000, "mp", 100, 0.8, 9997.9 / 9999.0);
      testFunc(10000, "mp", 100, 0.8, 0.1);

      testFunc(10000, "kpm", 100, 0.8, 0.1);
      testFunc(10000, "kmp", 100, 0.8, 0.1);
      testFunc(10000, "pkm", 100, 0.8, 0.1);
      testFunc(10000, "pmk", 100, 0.8, 0.1);
      testFunc(10000, "mkp", 100, 0.8, 0.1);
      testFunc(10000, "mpk", 100, 0.8, 0.1);
    });
  });
}
