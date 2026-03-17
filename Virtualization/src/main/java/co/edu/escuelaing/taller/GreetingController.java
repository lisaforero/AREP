package co.edu.escuelaing.taller;

@RestController
public class GreetingController {

    private static final String template = "Hello, %s!";

    @GetMapping("/greeting")
    public String greeting(@RequestParam(value = "name", defaultValue = "World") String name) {
        return "Hello " + name + "! You are using a concurrent framework.";
    }
}