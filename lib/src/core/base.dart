import 'dart:ffi' as ffi;

abstract class NativeObject<T extends ffi.NativeType> implements ffi.Finalizable {
  ffi.Pointer<T> ptr;

  void dispose();

  NativeObject(this.ptr);
}

abstract class LLAMAStruct<T extends ffi.Struct> extends NativeObject<T> {
  T get ref;

  LLAMAStruct(super.ptr);
}

abstract class LLAMAClass<T extends ffi.Opaque> extends NativeObject<T> {
  LLAMAClass(super.ptr);
}
