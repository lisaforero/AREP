package co.edu.escuelaing.taller;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
public @interface RequestParam {
    String value();
    String defaultValue() default "";
}