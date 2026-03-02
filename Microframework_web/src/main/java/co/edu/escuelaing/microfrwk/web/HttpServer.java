package co.edu.escuelaing.microfrwk.web;

import java.net.*;
import java.util.Map;
import java.util.HashMap;
import java.io.*;

public class HttpServer {

    static Map<String, WebMethod> endPoints = new HashMap<>();
    private static String staticFilesLocation = "/webroot";

    public static void main(String[] args) throws IOException, URISyntaxException {
        ServerSocket serverSocket = null;
        try {
            serverSocket = new ServerSocket(8080);
        } catch (IOException e) {
            System.err.println("Could not listen on port.");
            System.exit(1);
        }

        boolean running = true;
        while (running) {
            Socket clientSocket = null;
            try {
                System.out.println("\nReady to receive ...");
                clientSocket = serverSocket.accept();
            } catch (IOException e) {
                System.err.println("Accept failed.");
                System.exit(1);
            }

            PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
            BufferedOutputStream dataOut = new BufferedOutputStream(clientSocket.getOutputStream());
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            
            String inputLine, outputLine = "";
            boolean isFirstLine = true;
            String reqpath = "";
            String requery = "";

            while ((inputLine = in.readLine()) != null) {
                System.out.println("Received: " + inputLine);
                if (isFirstLine) {
                    String[] flTokens = inputLine.split(" ");
                    if (flTokens.length > 1) {
                        URI uripath = new URI(flTokens[1]);
                        reqpath = uripath.getPath();
                        requery = uripath.getQuery();
                        System.out.println("Path: " + reqpath);
                    }
                    isFirstLine = false;
                }
                if (!in.ready()) break;
            }

            WebMethod currentWm = endPoints.get(reqpath);
            if (currentWm == null && reqpath.startsWith("/app/")) {
                currentWm = endPoints.get(reqpath.substring(4)); 
            }
            
            if (currentWm != null) {
                outputLine = "HTTP/1.1 200 OK\r\n"
                    + "Content-Type: text/html\r\n"
                    + "\r\n"
                    + "<!DOCTYPE html>"
                    + "<html>"
                    + "<head>"
                    + "<meta charset=\"UTF-8\">"
                    + "<title>Backend Service</title>\n"
                    + "</head>"
                    + "<body>"
                    + currentWm.execute(new HttpRequest(requery), new HttpResponse())
                    + "</body>"
                    + "</html>";
                out.println(outputLine);
            }
            else {
                serveStaticFile(reqpath, out, dataOut);
            }

            out.close();
            in.close();
            clientSocket.close();
        }
        serverSocket.close();
    }

    private static void serveStaticFile(String reqpath, PrintWriter out, BufferedOutputStream dataOut) throws IOException {
        String fullResourcePath = staticFilesLocation + reqpath;

        try (InputStream is = HttpServer.class.getResourceAsStream(fullResourcePath)) {
            if (is != null) {
                byte[] fileBytes = is.readAllBytes();
                String contentType = getContentType(reqpath);

                String header = "HTTP/1.1 200 OK\r\n"
                              + "Content-Type: " + contentType + "\r\n"
                              + "Content-Length: " + fileBytes.length + "\r\n"
                              + "\r\n";
                
                dataOut.write(header.getBytes());
                
                dataOut.write(fileBytes);
                dataOut.flush();
            } else {
                out.println("HTTP/1.1 404 Not Found\r\n\r\n<h1>404: File not found</h1>");
            }
        }
    }

    private static String getContentType(String path) {
        if (path.endsWith(".html")) return "text/html";
        if (path.endsWith(".png")) return "image/png";
        if (path.endsWith(".jpg")) return "image/jpeg";
        return "text/plain";
    }

    public static void get(String path, WebMethod wm) {
        endPoints.put(path, wm);
    }

    public static void staticfiles(String path) {
        staticFilesLocation = path;
    }
}