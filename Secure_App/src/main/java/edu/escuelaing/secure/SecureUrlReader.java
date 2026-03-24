package edu.escuelaing.secure;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.security.KeyStore;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;

public class SecureUrlReader {

    public static void main(String[] args) throws Exception {
        File trustStoreFile = new File("keystores/myTrustStore");
        char[] trustStorePassword = "123456".toCharArray();

        KeyStore trustStore = KeyStore.getInstance(KeyStore.getDefaultType());
        trustStore.load(new FileInputStream(trustStoreFile), trustStorePassword);

        TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
        tmf.init(trustStore);

        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(null, tmf.getTrustManagers(), null);
        SSLContext.setDefault(sslContext);

        readURL("https://localhost:5000/hello"); // it should work
        readURL("https://www.google.com"); // it should fail
    }

    private static void readURL(String urlString) {
        try {
            URL url = new URL(urlString);
            BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));
            String inputLine;
            System.out.println("Content of " + urlString + ":");
            while ((inputLine = in.readLine()) != null) {
                System.out.println(inputLine);
            }
            in.close();
        } catch (Exception e) {
            System.out.println("Error reading " + urlString + ": " + e.getMessage());
        }
    }
}