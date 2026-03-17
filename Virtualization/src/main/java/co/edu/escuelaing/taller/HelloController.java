package co.edu.escuelaing.taller;

@RestController
public class HelloController {

    @GetMapping("/status")
	public String index() {
		return "Greetings from my own Concurrent Framework!";
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

	@GetMapping("/shutdown")
	public String shutdown() {
		new Thread(() -> {
			try {
				Thread.sleep(1000);
				System.exit(0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}).start();

		return "The server is shutting down...";
	}
}