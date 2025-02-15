// Copyright (c) 2023, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:io';
import 'package:native_assets_cli/code_assets_builder.dart';
import 'package:path/path.dart' as p;
import 'package:logging/logging.dart';
import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

/// Implements the protocol from `package:native_assets_cli` by building
/// the C code in `src/` and reporting what native assets it built.
void main(List<String> args) async {
  await build(args, _builder);
}

Future<void> _builder(BuildInput input, BuildOutputBuilder output) async {
  final env = Platform.environment;

  final packageName = input.packageName;
  final cbuilder = CMakeBuilder.create(
    name: packageName,
    sourceDir: "src/",
    targets: [
      // '',
    ],
    defines: {
      'LLAMA_BUILD_TESTS': 'OFF',
      'LLAMA_BUILD_EXAMPLES': 'OFF',
      'LLAMA_BUILD_SERVER': 'OFF',
      // 'CMAKE_INSTALL_PREFIX': input.outputDirectory.resolve('install').toFilePath(),
    },
  );

  await cbuilder.run(
    input: input,
    output: output,
    logger: Logger("")
      ..level = Level.ALL
      ..onRecord.listen((record) => print(record.message)),
  );

  output.assets.code.addAll(
    [
      CodeAsset(
        package: input.packageName,
        name: "$packageName.dart",
        file: Uri.parse('${input.outputDirectory.resolve('bin/libllama.dylib')}'),
        linkMode: DynamicLoadingBundled(),
        os: input.config.code.targetOS,
        architecture: input.config.code.targetArchitecture,
      ),
      CodeAsset(
        package: input.packageName,
        name: "ggml.dart",
        file: Uri.parse('${input.outputDirectory.resolve('bin/libggml.dylib')}'),
        linkMode: DynamicLoadingBundled(),
        os: input.config.code.targetOS,
        architecture: input.config.code.targetArchitecture,
      ),
    ],
  );
}
