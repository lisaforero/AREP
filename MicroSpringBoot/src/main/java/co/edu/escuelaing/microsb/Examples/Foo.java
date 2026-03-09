package co.edu.escuelaing.microsb.Examples;



public class Foo {
    @Test public static void m1() { }

    public static void m2() { }

    @Test public static void m3() {
        throw new RuntimeException("Boom");
    }

    public static void m4() { }

    @Test public static void m5() { }

    public static void m6() { }

    @Test public static void m7() {
        throw new RuntimeException("Crash");
    }

    public static void m8() { }
}

//java -cp target/classes co.edu.escuelaing.Examples.RunTests co.edu.escuelaing.Examples.Foo