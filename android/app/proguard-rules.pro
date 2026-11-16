# UniFFI's bindings reach the engine through JNA, which finds native methods
# and structures by reflection: keep them, and JNA itself, unrenamed.
-keep class com.sun.jna.** { *; }
-keepclassmembers class * extends com.sun.jna.** { public *; }
-keep class pl.scrablozaur.engine.** { *; }
-dontwarn java.awt.**
