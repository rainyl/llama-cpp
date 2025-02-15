import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'enums.dart';
import 'context.dart';
import '../g/llama.g.dart' as llama;

class ThreadPool extends LLAMAClass<llama.ggml_threadpool> {
  ThreadPool(super.ptr);

  @override
  void dispose() {
    // TODO: implement dispose
  }
}

void attachThreadPool(Context ctx, ThreadPool pool, ThreadPool poolBatch) =>
    llama.llama_attach_threadpool(ctx.ptr, pool.ptr, poolBatch.ptr);

void detachThreadPool(Context ctx) => llama.llama_detach_threadpool(ctx.ptr);
