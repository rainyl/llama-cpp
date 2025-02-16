import '../g/llama.g.dart' as C;
import 'base.dart';
import 'context.dart';

class ThreadPool extends LLAMAClass<C.ggml_threadpool> {
  ThreadPool(super.ptr);

  @override
  void dispose() {}
}

void attachThreadPool(Context ctx, ThreadPool pool, ThreadPool poolBatch) =>
    C.llama_attach_threadpool(ctx.ptr, pool.ptr, poolBatch.ptr);

void detachThreadPool(Context ctx) => C.llama_detach_threadpool(ctx.ptr);
