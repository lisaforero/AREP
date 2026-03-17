package co.edu.escuelaing.taller.web;

import java.net.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class HttpServer {
    static Map<String, WebMethod> endPoints = new HashMap<>();
    private static String staticFilesLocation = "/webroot";

    private static final ExecutorService threadPool = Executors.newFixedThreadPool(10);
    private static ServerSocket serverSocket;
    private static boolean running = true;

    public static void main(String[] args) throws IOException {

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[SERVER] Starting the elegant shutdown...");
            running = false;
            try {
                threadPool.shutdown(); 
                if (!threadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                    threadPool.shutdownNow(); 
                }
                serverSocket.close();
                System.out.println("[SERVER] Goodbye!");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }));

        try {
            serverSocket = new ServerSocket(6000);
        } catch (IOException e) {
            System.err.println("Could not listen on port 6000.");
            System.exit(1);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutting down the server...");
            stopServer();
        }));

        System.out.println("Ready to receive on port 8080...");
        
        while (running) {
            try {
                Socket clientSocket = serverSocket.accept();
                threadPool.execute(() -> handleRequest(clientSocket));
            } catch (IOException e) {
                if (running) System.err.println("Accept failed.");
            }
        }
    }

    private static void handleRequest(Socket clientSocket) {
        try (
            PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
            BufferedOutputStream dataOut = new BufferedOutputStream(clientSocket.getOutputStream());
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))
        ) {
            String inputLine;
            boolean isFirstLine = true;
            String reqpath = "";
            String requery = "";
            
            while ((inputLine = in.readLine()) != null) {
                if (isFirstLine) {
                    String[] flTokens = inputLine.split(" ");
                    if (flTokens.length > 1) {
                        URI uripath = new URI(flTokens[1]);
                        reqpath = uripath.getPath();
                        requery = uripath.getQuery();
                    }
                    isFirstLine = false;
                }
                if (!in.ready()) break;
            }
            
            WebMethod currentWm = endPoints.get(reqpath);
            if (currentWm != null) {
                String responseBody = currentWm.execute(new HttpRequest(requery), new HttpResponse());
                String outputLine = "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: text/html\r\n"
                        + "\r\n"
                        + responseBody;
                out.println(outputLine);
            }else if (reqpath.equals("/")){
                File folder = new File(HttpServer.class.getClassLoader().getResource("webroot").getPath());
                String response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
                                + "<html><body><h1>Static files directory</h1><ul>";
                
                for (File file : folder.listFiles()) {
                    String name = file.getName();
                    response += "<li><a href='/" + name + "'>" + name + "</a></li>";
                }
                
                response += "</ul></body></html>";
                out.println(response);
                        
            } else {
                serveStaticFile(reqpath, out, dataOut);
            }
            clientSocket.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void serveStaticFile(String reqpath, PrintWriter out, BufferedOutputStream dataOut) throws IOException {
        String path = (reqpath.equals("/")) ? "/index.html" : reqpath;
        String fullPath = "/webroot" + path;

        try (InputStream is = HttpServer.class.getResourceAsStream(fullPath)) {
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
        if (path.endsWith(".css")) return "text/css";
        if (path.endsWith(".js")) return "application/javascript";
        if (path.endsWith(".png")) return "image/png";
        if (path.endsWith(".jpg")) return "image/jpeg";
        return "text/plain";
    }

    public static void staticfiles(String path) {
        staticFilesLocation = path;
    }

    public static void stopServer() {
        running = false;
        try {
            threadPool.shutdown();
            if (!threadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                threadPool.shutdownNow();
            }
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
            }
        } catch (Exception e) {
            System.err.println("Error during shutdown: " + e.getMessage());
        }
    }

    public static void get(String path, WebMethod wm) {
        endPoints.put(path, wm);
    }
}