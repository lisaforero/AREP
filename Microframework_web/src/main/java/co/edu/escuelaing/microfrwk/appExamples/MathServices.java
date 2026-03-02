package co.edu.escuelaing.microfrwk.appExamples;

import java.io.IOException;
import java.net.URISyntaxException;

import co.edu.escuelaing.microfrwk.web.HttpServer;

public class MathServices {

    public static void main(String[] args) throws IOException, URISyntaxException {
        
        HttpServer.get("/pi", (req, res) -> "PI = " + Math.PI);

        HttpServer.get("/hello", (req, res) -> "Hello " + req.getValues("name"));

        HttpServer.get("/e", (req, resp) -> getEuler());
    
        HttpServer.main(args);
    }
    
    private static String getEuler() {
        return "e = " + Math.E;
    }
}