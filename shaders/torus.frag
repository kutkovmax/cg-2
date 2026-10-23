#version 450

// Входные данные из вершинного шейдера (Vertex Shader)
layout(location = 0) in vec3 fragNormal; // Нормаль интерполированная для этого пикселя
layout(location = 1) in vec3 fragPos;    // Координаты пикселя в мировом пространстве
layout(location = 2) in vec3 fragColor;  // Итоговый базовый цвет пикселя (процедурный * цвет из ImGui)

// Выходные данные - итоговый цвет пикселя на экране
layout(location = 0) out vec4 outColor;

void main() {
    // Направление на вымышленный источник света
    vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
    // Нормализуем вектор нормали (гарантируем, что его длина равна 1)
    vec3 norm = normalize(fragNormal);

    // 1. Ambient (Фоновое освещение) - объект никогда не бывает абсолютно черным
    float ambient = 0.18;
    
    // 2. Diffuse (Рассеянное освещение) - чем прямее свет падает на нормаль, тем ярче
    // dot() - скалярное произведение. Если угол больше 90 градусов (свет сзади), max обнулит его до 0.0
    float diff = max(dot(norm, lightDir), 0.0);

    // 3. Specular (Блики) - модель Блинна-Фонга (Blinn-Phong)
    // viewDir - направление от пикселя к камере (камера у нас на 0, -4, 2)
    vec3 viewDir = normalize(vec3(0.0, -4.0, 2.0) - fragPos);
    // halfDir - вектор-биссектриса между светом и взглядом
    vec3 halfDir = normalize(lightDir + viewDir);
    // Если нормаль почти совпадает с halfDir, мы видим яркий блик (возводим в 48-ю степень для "остроты" блика)
    float spec = pow(max(dot(norm, halfDir), 0.0), 48.0) * 0.4;

    // Смешиваем свет (фон + рассеянный + блик) и умножаем на цвет фигуры
    vec3 result = (ambient + diff + spec) * fragColor;
    
    // Выдаем финальный цвет с прозрачностью 1.0 (полностью непрозрачный)
    outColor = vec4(result, 1.0);
}
