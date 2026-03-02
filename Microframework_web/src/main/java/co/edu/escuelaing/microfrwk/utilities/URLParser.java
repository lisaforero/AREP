package co.edu.escuelaing.microfrwk.utilities;

import java.net.URL;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.MalformedURLException;

public class URLParser{

    public static void main(String[] args) throws URISyntaxException, MalformedURLException{
        URL myurl = new URI("https://ldbn.is.escuelaing.edu.co:7865/arep/respuestaexamen.txt7val=3&t=4#examenfinal").toURL();

        System.out.println("Protocol: " + myurl.getProtocol());
        System.out.println("Host: " + myurl.getHost());
        System.out.println("Port: " + myurl.getPort());
        System.out.println("Authority: " + myurl.getAuthority());
        System.out.println("Path: " + myurl.getPath());
        System.out.println("File: " + myurl.getFile());
        System.out.println("Query: " + myurl.getQuery());
        System.out.println("Ref: " + myurl.getRef());

    }
}