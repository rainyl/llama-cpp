import 'package:test/test.dart';

class IterableCloseToMatcher extends Matcher {
  final Iterable<num> expected;
  final double epsilon;

  IterableCloseToMatcher(this.expected, this.epsilon);

  @override
  Description describe(Description description) {
    return description
        .addDescriptionOf(expected)
        .add(' with epsilon=$epsilon');
  }

  @override
  bool matches(dynamic item, Map matchState) {
    if (item is! Iterable<num>) {
      return false;
    }

    final actual = item;
    if (actual.length != expected.length) {
      matchState['mismatch'] = 'Length mismatch: expected ${expected.length}, got ${actual.length}';
      return false;
    }

    for (int i = 0; i < actual.length; i++) {
      if ((actual.elementAt(i) - expected.elementAt(i)).abs() >= epsilon) {
        matchState['mismatch'] =
            'Mismatch at index $i: expected ${expected.elementAt(i)}, got ${actual.elementAt(i)}';
        return false;
      }
    }

    return true;
  }

  @override
  Description describeMismatch(dynamic item, Description mismatchDescription, Map matchState, bool verbose) {
    if (matchState.containsKey('mismatch')) {
      return mismatchDescription.add(matchState['mismatch'].toString());
    }
    return mismatchDescription;
  }
}

Matcher iterableCloseTo(Iterable<num> expected, double epsilon) {
  return IterableCloseToMatcher(expected, epsilon);
}
