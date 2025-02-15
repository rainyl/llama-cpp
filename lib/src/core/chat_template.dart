import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'base.dart';
import '../g/llama.g.dart' as llama;

/// used in chat template
class ChatMessage extends LLAMAStruct<llama.llama_chat_message> {
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ChatMessage(super.ptr, {bool attach = true}) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory ChatMessage.init({String? role, String? content}) {
    final ptr = calloc<llama.llama_chat_message>();
    ptr.ref.role = role == null ? ffi.nullptr : role.toNativeUtf8().cast();
    ptr.ref.content = content == null ? ffi.nullptr : content.toNativeUtf8().cast();
    return ChatMessage(ptr);
  }

  String get role => ref.role.cast<Utf8>().toDartString();
  set role(String value) {
    ref.role = value.toNativeUtf8().cast();
  }

  String get content => ref.content.cast<Utf8>().toDartString();
  set content(String value) {
    ref.content = value.toNativeUtf8().cast();
  }

  static int bytesCount(String msg) {
    final p = msg.toNativeUtf8();
    final n = p.length;
    calloc.free(p);
    return n;
  }

  int selfBytesCount() {
    return bytesCount(role) + bytesCount(content);
  }

  @override
  void dispose() {
    calloc.free(ref.role);
    calloc.free(ref.content);
    finalizer.detach(this);
    calloc.free(ptr);
  }

  @override
  llama.llama_chat_message get ref => ptr.ref;
}

/// Apply chat template. Inspired by hf apply_chat_template() on python.
/// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
/// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
/// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
/// @param chat Pointer to a list of multiple llama_chat_message
/// @param n_msg Number of llama_chat_message in this chat
/// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
/// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
/// @param length The size of the allocated buffer
/// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
(int, String) applyChatTemplate(String tmplate, List<ChatMessage> chat, {bool addAss = true}) {
  final cTmpl = tmplate.toNativeUtf8().cast<ffi.Char>();
  final pChat = calloc<llama.llama_chat_message>(chat.length);
  int nBuf = 0;
  for (var i = 0; i < chat.length; i++) {
    pChat[i] = chat[i].ref;
    nBuf += chat[i].selfBytesCount();
  }
  int totalBufLength = 0;
  final pBuf = calloc<ffi.Char>(nBuf);
  totalBufLength = llama.llama_chat_apply_template(cTmpl, pChat, chat.length, addAss, pBuf, 2 * nBuf);  // TODO: memory issue
  return (totalBufLength, pBuf.cast<Utf8>().toDartString());
}

/// Get list of built-in chat templates
(int, List<String>) builtinChatTemplates() {
  const n = 32;
  final pbuf = calloc<ffi.Pointer<ffi.Char>>(n);
  final ret = llama.llama_chat_builtin_templates(pbuf, n);
  final templates = List.generate(n, (i) => pbuf[i].cast<Utf8>().toDartString());  // TODO: memory issue
  calloc.free(pbuf);
  return (ret, templates);
}
