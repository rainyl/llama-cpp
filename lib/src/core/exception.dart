class LLAMAException implements Exception {
  LLAMAException(this.message);

  final String message;

  @override
  String toString() => 'LLAMAException: $message';
}
