use anyhow::Result;

pub struct FaceQuality;

impl FaceQuality {
    pub fn new() -> Self {
        Self
    }

    pub fn assess(
        &self,
        detector_score: f32,
        face_area: f32,
        blur_score: f32,
        yaw: f32,
        pitch: f32,
        roll: f32,
    ) -> f32 {
        let face_area_score = (face_area / 10000.0).min(1.0);

        let pose_score = {
            let yaw_dev = (yaw.abs() - 15.0).max(0.0) / 45.0;
            let pitch_dev = (pitch.abs() - 15.0).max(0.0) / 45.0;
            let roll_dev = (roll.abs() - 15.0).max(0.0) / 45.0;
            1.0 - (yaw_dev + pitch_dev + roll_dev) / 3.0
        };

        let quality = 0.30 * detector_score
            + 0.25 * face_area_score
            + 0.25 * blur_score
            + 0.20 * pose_score;

        quality.max(0.0).min(1.0)
    }

    pub fn is_acceptable(&self, quality: f32) -> bool {
        quality >= 0.45
    }
}