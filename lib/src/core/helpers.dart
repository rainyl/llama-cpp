import 'package:ffi/ffi.dart';

import 'enums.dart';
import '../g/llama.g.dart' as llama;

void numaInit(GGMLNumaStrategy strategy) => llama.llama_numa_init(strategy);

int timeUs() => llama.llama_time_us();

int maxDevices() => llama.llama_max_devices();

bool supportsMmap() => llama.llama_supports_mmap();

bool supportsMLock() => llama.llama_supports_mlock();

bool supportsGpuOffload() => llama.llama_supports_gpu_offload();

bool supportsRPC() => llama.llama_supports_rpc();

/// @details Build a split GGUF final path for this chunk.
/// llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
/// Returns the split_path length.
// external int llama_split_path(
//   ffi.Pointer<ffi.Char> split_path,
//   int maxlen,
//   ffi.Pointer<ffi.Char> path_prefix,
//   int split_no,
//   int split_count,
// );

/// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
/// llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
/// Returns the split_prefix length.
// external int llama_split_prefix(
//   ffi.Pointer<ffi.Char> split_prefix,
//   int maxlen,
//   ffi.Pointer<ffi.Char> split_path,
//   int split_no,
//   int split_count,
// );

/// Print system information
String printSystemInfo() => llama.llama_print_system_info().cast<Utf8>().toDartString();

/// Set callback for all future logging events.
/// If this is not called, or NULL is supplied, everything is output on stderr.
// @ffi.Native<ffi.Void Function(ggml_log_callback, ffi.Pointer<ffi.Void>)>()
// external void llama_log_set(
//   ggml_log_callback log_callback,
//   ffi.Pointer<ffi.Void> user_data,
// );