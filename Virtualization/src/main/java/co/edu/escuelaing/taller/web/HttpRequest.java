package co.edu.escuelaing.taller.web;

import java.util.HashMap;
import java.util.Map;

public class HttpRequest {

    private Map<String, String> queryParams = new HashMap<>();

    public HttpRequest(String query) {
        if (query != null && !query.isEmpty()) {
            String[] pairs = query.split("&");
            for (String pair : pairs) {
                String[] parts = pair.split("=");
                    if (parts.length > 1) queryParams.put(parts[0], parts[1]);
            }
        }
    }

    public String getValues(String key) {
        return queryParams.getOrDefault(key, "");
    }
}