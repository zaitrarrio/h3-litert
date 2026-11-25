// RGBA bytes ↔ CGImage: frames out of the pipeline for `Image(decorative:scale:)`,
// and a picked photo into a `[3, 1, H, W]` keyframe at the canvas size.

import CoreGraphics
import Foundation
import H3LiteRTCore
#if canImport(UIKit)
import UIKit
#endif

enum FrameImage {
    /// One frame of a `Frames` event or a `Video` as a `CGImage` (copies the bytes).
    static func make(rgba: ArraySlice<UInt8>, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0, rgba.count == width * height * 4 else { return nil }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    static func images(of frames: Frames) -> [CGImage] {
        (0..<frames.count).compactMap { make(rgba: frames.frame($0), width: frames.width, height: frames.height) }
    }

    /// Draw `image` aspect-filled (centre-cropped) into a `width × height` RGBA buffer.
    static func rgba(of image: CGImage, width: Int, height: Int) -> [UInt8]? {
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info.rawValue
            ) else { return false }
            let sw = Double(image.width), sh = Double(image.height)
            let scale = max(Double(width) / sw, Double(height) / sh)
            let dw = sw * scale, dh = sh * scale
            let rect = CGRect(x: (Double(width) - dw) / 2, y: (Double(height) - dh) / 2, width: dw, height: dh)
            ctx.interpolationQuality = .high
            ctx.draw(image, in: rect)
            return true
        }
        guard ok else { return nil }
        // A bitmap context's first row in memory is the top of the drawn image (TODO verify on device:
        // if keyframes come out upside down, flip the rows here).
        return buf
    }

    /// A picked photo as an FL2VA keyframe: `[3, 1, height, width]` in `[0, 1]`.
    static func keyframe(anchor: String, image: CGImage, width: Int, height: Int) -> Keyframe? {
        guard let bytes = rgba(of: image, width: width, height: height) else { return nil }
        return Keyframe(anchor: anchor, pixels: Pixels.keyframePixels(rgba: bytes, width: width, height: height))
    }

    #if canImport(UIKit)
    static func cgImage(from data: Data) -> CGImage? {
        UIImage(data: data)?.cgImage
    }
    #endif
}
