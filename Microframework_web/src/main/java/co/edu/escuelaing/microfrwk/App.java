package co.edu.escuelaing.microfrwk;

import co.edu.escuelaing.microfrwk.web.HttpServer;
import java.io.IOException;
import java.net.URISyntaxException;

public class App {
    public static void main(String[] args) throws IOException, URISyntaxException {
        HttpServer.staticfiles("/webroot");

        HttpServer.get("/pi", (req, res) -> "PI: " + Math.PI);
        HttpServer.get("/hello", (req, res) -> "Hello " + req.getValues("name"));
        HttpServer.get("/e", (req, resp) -> getEuler());

        HttpServer.main(args);
    }

    private static String getEuler() {
        return "e = " + Math.E;
    }
}