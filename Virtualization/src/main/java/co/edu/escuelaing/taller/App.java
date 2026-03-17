package co.edu.escuelaing.taller;

import co.edu.escuelaing.taller.web.HttpServer;

import java.io.File;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.Map;

public class App {

    static Map<String, Method> controllerMethods = new HashMap<>();
    static Map<String, Object> controllerInstances = new HashMap<>();

    public static void main(String[] args) throws Exception {

        System.out.println("Loading components ...");

        File folder = new File(App.class.getClassLoader().getResource("").getPath());
        scanClasses(folder, "");

        for (String path : controllerMethods.keySet()) {

            HttpServer.get(path, (req, res) -> {

                try {

                    Method m = controllerMethods.get(path);
                    Object instance = controllerInstances.get(path);

                    Parameter[] parameters = m.getParameters();
                    Object[] values = new Object[parameters.length];

                    for (int i = 0; i < parameters.length; i++) {

                        if (parameters[i].isAnnotationPresent(RequestParam.class)) {

                            RequestParam rp = parameters[i].getAnnotation(RequestParam.class);

                            String paramName = rp.value();
                            String value = req.getValues(paramName);

                            if (value == null || value.equals("")) {
                                value = rp.defaultValue();
                            }

                            values[i] = value;
                        }
                    }

                    return (String) m.invoke(instance, values);

                } catch (Exception e) {
                    e.printStackTrace();
                    return "Error invoking method";
                }

            });
        }
        HttpServer.staticfiles("/webroot");
        System.out.println("Starting HTTP Server...");
        HttpServer.main(null);
    }

    private static void scanClasses(File folder, String packageName) throws Exception {

        for (File file : folder.listFiles()) {

            if (file.isDirectory()) {

                scanClasses(file, packageName + file.getName() + ".");

            } else if (file.getName().endsWith(".class")) {

                String className = packageName + file.getName().replace(".class", "");
                Class<?> c = Class.forName(className);

                if (c.isAnnotationPresent(RestController.class)) {

                    Object instance = c.getDeclaredConstructor().newInstance();

                    for (Method m : c.getDeclaredMethods()) {

                        if (m.isAnnotationPresent(GetMapping.class)) {

                            GetMapping a = m.getAnnotation(GetMapping.class);
                            String path = a.value();

                            controllerMethods.put(path, m);
                            controllerInstances.put(path, instance);

                            System.out.println("Loaded endpoint: " + path);
                        }
                    }
                }
            }
        }
    }
}
