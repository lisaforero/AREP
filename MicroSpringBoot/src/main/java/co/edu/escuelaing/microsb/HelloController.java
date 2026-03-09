package co.edu.escuelaing.microsb;

@RestController
public class HelloController {

    @GetMapping("/")
	public String index() {
		return "Greetings from Spring Boot!";
	}

    @GetMapping("/pi")
	public String getPI() {
		return "PI = " + Math.PI;
	}

    @GetMapping("/hello")
	public String hello() {
		return "Hello world!";
	}

	@GetMapping("/time")
	public static String time(){
		return java.time.LocalDateTime.now().toString();
	}
}