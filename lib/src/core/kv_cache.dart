import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:llama_cpp/src/core/context.dart';

import 'base.dart';
import 'enums.dart';
import '../g/llama.g.dart' as llama;

class ModelKvOverride extends LLAMAStruct<llama.llama_model_kv_override> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ModelKvOverride(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  ModelKvOverrideType get tag => ref.tag;
  set tag(ModelKvOverrideType value) => ref.tagAsInt = value.value;

  String get key => String.fromCharCodes(List.generate(128, (i) => ref.key[i]));
  ffi.Array get keyPtr => ref.key;

  int get valI64 => ref.unnamed.val_i64;
  double get valF64 => ref.unnamed.val_f64;
  bool get valBool => ref.unnamed.val_bool;
  String get valStr => String.fromCharCodes(List.generate(128, (i) => ref.unnamed.val_str[i]));

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  llama.llama_model_kv_override get ref => ptr.ref;
}

/// Information associated with an individual cell in the KV cache view.
class KvCacheViewCell extends LLAMAStruct<llama.llama_kv_cache_view_cell> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  KvCacheViewCell(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory KvCacheViewCell.fromNative(llama.llama_kv_cache_view_cell cell) {
    final p = calloc<llama.llama_kv_cache_view_cell>()..ref = cell;
    return KvCacheViewCell(p);
  }

  /// The position for this cell. Takes KV cache shifts into account.
  /// May be negative if the cell is not populated.
  int get pos => ref.pos;
  set pos(int value) => ref.pos = value;

  @override
  void dispose() {
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  llama.llama_kv_cache_view_cell get ref => ptr.ref;
}

/// An updateable view of the KV cache.
class KvCacheView extends LLAMAStruct<llama.llama_kv_cache_view> {
  static final finalizer = ffi.NativeFinalizer(llama.addresses.llama_kv_cache_view_free.cast());

  KvCacheView(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  // /// Create an empty KV cache view. (use only for debugging purposes)
  // factory KvCacheView.init(Context ctx) {
  //   final view = llama.llama_kv_cache_view_init(ctx.ptr, ctx.nSeqMax);
  //   final ptr = calloc<llama.llama_kv_cache_view>()..ref = view;
  //   return KvCacheView(ptr);
  // }

  /// Number of KV cache cells. This will be the same as the context size.
  int get nCells => ref.n_cells;
  set nCells(int value) => ref.n_cells = value;

  /// Maximum number of sequences that can exist in a cell. It's not an error
  /// if there are more sequences in a cell than this value, however they will
  /// not be visible in the view cells_sequences.
  int get nSeqMax => ref.n_seq_max;
  set nSeqMax(int value) => ref.n_seq_max = value;

  /// Number of tokens in the cache. For example, if there are two populated
  /// cells, the first with 1 sequence id in it and the second with 2 sequence
  /// ids then you'll have 3 tokens.
  int get tokenCount => ref.token_count;
  set tokenCount(int value) => ref.token_count = value;

  /// Number of populated cache cells.
  int get usedCells => ref.used_cells;
  set usedCells(int value) => ref.used_cells = value;

  /// Maximum contiguous empty slots in the cache.
  int get maxContiguous => ref.max_contiguous;
  set maxContiguous(int value) => ref.max_contiguous = value;

  /// Index to the start of the max_contiguous slot range. Can be negative
  /// when cache is full.
  int get maxContiguousIdx => ref.max_contiguous_idx;
  set maxContiguousIdx(int value) => ref.max_contiguous_idx = value;

  // ffi.Pointer<llama_kv_cache_view_cell> cells;
  /// Information for an individual cell.
  UnmodifiableListView<KvCacheViewCell> get cells =>
      UnmodifiableListView(List.generate(ref.n_cells, (i) => KvCacheViewCell.fromNative(ref.cells[i])));
  ffi.Pointer<llama.llama_kv_cache_view_cell> get cellsPtr => ref.cells;

  /// The sequences for each cell. There will be n_seq_max items per cell.
  Int32List get cellsSequences => ref.cells_sequences.asTypedList(ref.n_cells * ref.n_seq_max);
  ffi.Pointer<llama.llama_seq_id> get cellsSequencesPtr => ref.cells_sequences;

  /// Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
  void update(Context ctx) => llama.llama_kv_cache_view_update(ctx.ptr, ptr);

  /// Free a KV cache view. (use only for debugging purposes)
  @override
  void dispose() {
    finalizer.detach(this);
    llama.llama_kv_cache_view_free(ptr);
  }

  @override
  llama.llama_kv_cache_view get ref => ptr.ref;
}
