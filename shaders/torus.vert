#version 450

// Входные данные одной вершины из Vertex Buffer'а
layout(location = 0) in vec3 inPosition; // Позиция (X, Y, Z)
layout(location = 1) in vec3 inNormal;   // Нормаль
layout(location = 2) in vec3 inColor;    // Процедурный цвет (сгенерированный в коде C++)

// UBO (Uniform Buffer Object) - данные, которые мы присылаем с CPU каждый кадр
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;       // Матрица Модели (положение объекта в мире)
    mat4 view;        // Матрица Камеры (откуда мы смотрим)
    mat4 proj;        // Матрица Проекции (перспектива/ортографик)
    vec4 userColor;   // Цвет из интерфейса ImGui
} ubo;

// Выходные данные для передачи во фрагментный шейдер
// Эти значения будут автоматически плавно смешаны (интерполированы) для пикселей внутри треугольника
layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragPos;
layout(location = 2) out vec3 fragColor;

void main() {
    // 1. Сначала переводим локальную позицию вершины в мировое пространство
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz; // Отправляем позицию дальше для расчета света
    
    // 2. Пересчитываем нормаль. Inverse/transpose нужны, чтобы нормаль не искажалась при неравномерном масштабировании!
    fragNormal = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);
    
    // 3. Смешиваем процедурный цвет (inColor) с базовым цветом интерфейса (ubo.userColor) (Задания 4 и 5)
    fragColor = inColor * ubo.userColor.rgb;
    
    // 4. Финальный перевод точки на 2D-экран (применяем Камеру и Проекцию).
    // gl_Position - это встроенная переменная Vulkan/GLSL.
    gl_Position = ubo.proj * ubo.view * worldPos;
}
