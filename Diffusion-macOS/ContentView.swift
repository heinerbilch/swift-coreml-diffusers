//
//  ContentView.swift
//  Diffusion-macOS
//
//  Created by Cyril Zakka on 1/12/23.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import ImageIO
import OSLog
import Photos
import SwiftUI

private let logger = Logger(
    subsystem: "co.huggingface.diffusers",
    category: "ContentView"
)
// AppKit version that uses NSImage, NSSavePanel
struct ShareButtons: View {
    var image: CGImage
    var name: String

    var filename: String {
        name.replacingOccurrences(of: " ", with: "_")
    }

    func showSavePanel() -> URL? {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.png]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Save your image"
        savePanel.message = "Choose a folder and a name to store the image."
        savePanel.nameFieldLabel = "File name:"
        savePanel.nameFieldStringValue = filename

        let response = savePanel.runModal()
        return response == .OK ? savePanel.url : nil
    }

    func savePNG(cgImage: CGImage, path: URL) {
        let image = NSImage(cgImage: cgImage, size: .zero)
        let imageRepresentation = NSBitmapImageRep(
            data: image.tiffRepresentation!
        )
        guard
            let pngData = imageRepresentation?.representation(
                using: .png,
                properties: [:]
            )
        else {
            logger.notice("Error generating PNG data")
            return
        }
        do {
            try pngData.write(to: path)
        } catch {
            logger.notice("Error saving: \(error)")
        }
    }

    var body: some View {
        let imageView = Image(image, scale: 1, label: Text(name))
        HStack {
            ShareLink(
                item: imageView,
                preview: SharePreview(name, image: imageView)
            )
            Button {
                if let url = showSavePanel() {
                    savePNG(cgImage: image, path: url)
                    logger.info("Saving \(name)")
                }
            } label: {
                Label("Save…", systemImage: "square.and.arrow.down")
            }
        }
    }
}

struct ContentView: View {
    @StateObject var generation = GenerationContext()

    func photoAuthorizationStatus() -> PHAuthorizationStatus {
        return PHPhotoLibrary.authorizationStatus(for: .addOnly)
    }

    func requestPhotoLibraryAccess() {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) {
            authorizationStatus in
            switch authorizationStatus {
            case .limited:
                logger.debug("limited authorization granted")
            case .restricted:
                logger.debug("restricted")
            case .authorized:
                logger.debug("authorization granted")
            default:
                logger.warning("Unimplemented")
            }
        }
    }

    func toolbar() -> any View {
        if case .complete(let prompt, let cgImage, _, _) = generation.state,
            let cgImage = cgImage
        {
            let status = photoAuthorizationStatus().rawValue
            logger.debug("Photo Authorization Status: \(status)")
            if status == 0 {
                requestPhotoLibraryAccess()
            }
            // TODO: share seed too
            logger.info("Sharing \(prompt)")
            return ShareButtons(image: cgImage, name: prompt)
        } else {
            let prompt = DEFAULT_PROMPT
            let cgImage = NSImage(imageLiteralResourceName: "placeholder")
                .cgImage(forProposedRect: nil, context: nil, hints: nil)!
            return ShareButtons(image: cgImage, name: prompt)
        }
    }

    var body: some View {
        NavigationSplitView {
            ControlsView()
                .navigationSplitViewColumnWidth(min: 250, ideal: 300)
        } detail: {
            GeneratedImageView()
                .aspectRatio(contentMode: .fit)
                .frame(width: 512, height: 512)
                .cornerRadius(15)
                .toolbar {
                    AnyView(toolbar())
                }
        }
        .environmentObject(generation)
    }
}

#Preview {
    ContentView()
}
