package edu.escuelaing.secure.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class AuthController {

    @GetMapping("/secure-data")
    public String secureData() {
        return "This is a protected resource";
    }

    @GetMapping("/public-data")
    public String publicData() {
        return "This is a public resource";
    }
}