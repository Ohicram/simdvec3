#ifndef VEC3_H
#define VEC3_H

#include "commons.h"

#include <cmath>
#include <iostream>
#define SSE_VEC

#ifdef SSE_VEC
#include <immintrin.h>
namespace sse
{
    class vec3 {
    public:
        vec3() {
            e_sse = _mm_set1_ps(0.f);
        }

        vec3(real e0, real e1, real e2) {
            e_sse = _mm_set_ps(0.f, e2, e1, e0);
        }

        vec3(__m128 in_sse) : e_sse(in_sse) {}

        float x() const { return e[0]; }
        float y() const { return e[1]; }
        float z() const { return e[2]; }

        vec3 operator-() const { return vec3(_mm_mul_ps(e_sse, _mm_set1_ps(-1.f))); }
        float operator[](int i) const { return e[i]; }
        float& operator[](int i) { return e[i]; }

        vec3& operator+=(const vec3& v) {
            e_sse = _mm_add_ps(e_sse, v.e_sse);
            return *this;
        }

        vec3 operator+(const vec3& v) const
        {
            return vec3(_mm_add_ps(e_sse, v.e_sse));
        }

        vec3 operator-(const vec3& v) const
        {
            return vec3(_mm_sub_ps(e_sse, v.e_sse));
        }

        vec3 operator*(const vec3& v) const
        {
            return vec3(_mm_mul_ps(e_sse, v.e_sse));
        }

        vec3& operator*=(const float t) {
            e_sse = _mm_mul_ps(e_sse, _mm_set1_ps(t));
            return *this;
        }

        vec3& operator+=(const float t) {
            e_sse = _mm_add_ps(e_sse, _mm_set_ps(0.f, t, t, t));
            return *this;
        }


        vec3& operator/=(const real t) {
            e_sse = _mm_div_ps(e_sse, _mm_set1_ps(t));
            return *this;
        }


        float length_squared() const {
            //__m128 v = _mm_mul_ps(e_sse, e_sse);
            //__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0 4321 -> 3311
            //__m128 sums = _mm_add_ps(v, shuf);       // sums is now 7632
            //shuf = _mm_movehl_ps(shuf, sums); // high half -> low half (3311, 7632) -> (3211)
            //sums = _mm_add_ss(sums, shuf); 
           __m128 sums = _mm_dp_ps(e_sse, e_sse, 255);      // sums is now (7632+3211) -> (10 8 4 3)
            return _mm_cvtss_f32(sums);
        }

        float length() const {
           // __m128 v = _mm_mul_ps(e_sse, e_sse);
            //__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0 4321 -> 3311
           // __m128 sums = _mm_add_ps(v, shuf);       // sums is now 7632
           // shuf = _mm_movehl_ps(shuf, sums); // high half -> low half (3311, 7632) -> (3211)
           // sums = _mm_add_ss(sums, shuf);      // sums is now (7632+3211) -> (10 8 4 3)
            __m128 l = _mm_sqrt_ps(_mm_dp_ps(e_sse, e_sse, 255));
            return _mm_cvtss_f32(l);
        }

        inline static vec3 random() {
            return vec3(random_real(), random_real(), random_real());
        }

        inline static vec3 random(real min, real max) {
            return vec3(random_real(min, max), random_real(min, max), random_real(min, max));
        }

        vec3& normalized()
        {
            (*this) /= this->length();
            return *this;
        }

        bool near_zero() const {
            // Return true if the vector is close to zero in all dimensions.
            const auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        union
        {
            __m128 e_sse;
            float e[4];
        };
    };

    sse::vec3 random_in_unit_disk() {
        real cos_theta = random_real(-1, 1);
        real sin_theta = sqrt(1 - cos_theta * cos_theta) * random_real(-1.0, 1.0);
        return vec3(cos_theta, sin_theta, 0.0);
    }
    // vec3 Utility Functions
    inline std::ostream& operator<<(std::ostream& out, const sse::vec3& v) {
        return out << v[0] << ' ' << v[1] << ' ' << v[2];
    }

    inline sse::vec3 operator*(float t, const sse::vec3& v) {
        sse::vec3 w = v;
        return w*=t;
    }

    inline sse::vec3 operator*(const sse::vec3& v, real t) {
        return t * v;
    }

    inline sse::vec3 operator+(real t, const sse::vec3& v) {
        sse::vec3 w = v;
        return w += t;
    }

    inline sse::vec3 operator+(const sse::vec3& v, real t) {
        return t + v;
    }

    inline sse::vec3 operator/(sse::vec3 v, real t) {
        sse::vec3 w = v;
        return w /= t;
    }

    inline float dot(const sse::vec3& u, const sse::vec3& v) {
        return _mm_cvtss_f32(_mm_dp_ps(u.e_sse, v.e_sse, 255));
    }

    inline sse::vec3 cross(const sse::vec3& a, const sse::vec3& b) {
        __m128 a_yzx = _mm_shuffle_ps(a.e_sse, a.e_sse, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 a_zxy = _mm_shuffle_ps(a.e_sse, a.e_sse, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 b_zxy = _mm_shuffle_ps(b.e_sse, b.e_sse, _MM_SHUFFLE(3, 1, 0, 1));
        __m128 b_yzx = _mm_shuffle_ps(b.e_sse, b.e_sse, _MM_SHUFFLE(3, 0, 2, 1));
        return sse::vec3(_mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx)));

    }

    inline vec3 unit_vector(sse::vec3 v) {
        return v / v.length();
    }

    sse::vec3 random_in_unit_sphere() {
        return sse::vec3::random(-1, 1).normalized() * random_real(0.0001, 1);
    }

    sse::vec3 random_unit_vector() {
        return vec3::random(-1, 1).normalized();
    }

    sse::vec3 random_in_hemisphere(const sse::vec3& normal) {
        vec3 in_unit_sphere = random_in_unit_sphere();
        if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
            return in_unit_sphere;
        else
            return -in_unit_sphere;
    }



    sse::vec3 reflect(const sse::vec3& v, const sse::vec3& n) {
        return v - 2 * dot(v, n) * n;
    }

    sse::vec3 refract(const sse::vec3& inc_ray, const sse::vec3& n, real refraction_ratio) {
        real cos_theta = fmin(dot(-inc_ray, n), 1.0);
        // t_ray = transmitted ray
        sse::vec3 t_ray_perpendicular = refraction_ratio * (inc_ray + cos_theta * n); // We simply subtract to i its parallel component to (n) that is its projection on n and multiply by ratio
        sse::vec3 t_ray_parallel = -sqrt(fabs(1.0 - t_ray_perpendicular.length_squared())) * n;
        return t_ray_parallel + t_ray_perpendicular;
    }
};
using namespace sse;

#else
    class vec3 {
    public:
        vec3() : e{ 0,0,0 } {}
        vec3(real e0, real e1, real e2) : e{ e0, e1, e2 } {}

        real x() const { return e[0]; }
        real y() const { return e[1]; }
        real z() const { return e[2]; }

        vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        real operator[](int i) const { return e[i]; }
        real& operator[](int i) { return e[i]; }

        vec3& operator+=(const vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        vec3& operator*=(const real t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        vec3& operator+=(const real t) {
            e[0] += t;
            e[1] += t;
            e[2] += t;
            return *this;
        }

        vec3& operator/=(const real t) {
            return *this *= 1 / t;
        }

        real length() const {
            return sqrt(length_squared());
        }

        real length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        inline static vec3 random() {
            return vec3(random_real(), random_real(), random_real());
        }

        inline static vec3 random(real min, real max) {
            return vec3(random_real(min, max), random_real(min, max), random_real(min, max));
        }

        vec3& normalized()
        {
            (*this) /= this->length();
            return *this;
        }

        bool near_zero() const {
            // Return true if the vector is close to zero in all dimensions.
            const auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

    public:
        real e[3];
    };

    // vec3 Utility Functions
    inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
        return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
    }

    inline vec3 operator+(const vec3& u, const vec3& v) {
        return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
    }

    inline vec3 operator-(const vec3& u, const vec3& v) {
        return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
    }

    inline vec3 operator*(const vec3& u, const vec3& v) {
        return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
    }

    inline vec3 operator*(real t, const vec3& v) {
        return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    inline vec3 operator*(const vec3& v, real t) {
        return t * v;
    }

    inline vec3 operator+(real t, const vec3& v) {
        return vec3(t + v.e[0], t + v.e[1], t + v.e[2]);
    }

    inline vec3 operator+(const vec3& v, real t) {
        return t + v;
    }

    inline vec3 operator/(vec3 v, real t) {
        return (1 / t) * v;
    }

    inline real dot(const vec3& u, const vec3& v) {
        return u.e[0] * v.e[0]
            + u.e[1] * v.e[1]
            + u.e[2] * v.e[2];
    }

    inline vec3 cross(const vec3& u, const vec3& v) {
        return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
            u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }

    inline vec3 unit_vector(vec3 v) {
        return v / v.length();
    }

    vec3 random_in_unit_sphere() {
        return vec3::random(-1, 1).normalized() * random_real(0.0001, 1);
    }

    vec3 random_unit_vector() {
        return vec3::random(-1, 1).normalized();
    }

    vec3 random_in_hemisphere(const vec3& normal) {
        vec3 in_unit_sphere = random_in_unit_sphere();
        if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
            return in_unit_sphere;
        else
            return -in_unit_sphere;
    }

    vec3 random_in_unit_disk() {
        real cos_theta = random_real(-1, 1);
        real sin_theta = sqrt(1 - cos_theta * cos_theta) * random_real(-1.0, 1.0);
        return vec3(cos_theta, sin_theta, 0.0);
    }

    vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2 * dot(v, n) * n;
    }

    //  (i) (n) (r)
    //    \  |  /
    //     \ | /
    //      \|/
    // ---------------
    //        \
    //         \ (t)   
    // i = incoming ray | n = normal | r = reflected ray | t = transmitted ray
    vec3 refract(const vec3& inc_ray, const vec3& n, real refraction_ratio) {
        real cos_theta = fmin(dot(-inc_ray, n), 1.0);
        // t_ray = transmitted ray
        vec3 t_ray_perpendicular = refraction_ratio * (inc_ray + cos_theta * n); // We simply subtract to i its parallel component to (n) that is its projection on n and multiply by ratio
        vec3 t_ray_parallel = -sqrt(fabs(1.0 - t_ray_perpendicular.length_squared())) * n;
        return t_ray_parallel + t_ray_perpendicular;
    }
#endif // SSE
// Type aliases for vec3
//using namespace sse;
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

#endif