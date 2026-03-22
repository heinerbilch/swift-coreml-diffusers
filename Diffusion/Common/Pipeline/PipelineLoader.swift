//
//  PipelineLoader.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//


import CoreML
import Combine

import ZIPFoundation
import StableDiffusion
import OSLog

class PipelineLoader {
    private let logger = Logger(subsystem: "com.huggingface.swift-coreml-diffusers", category: "PipelineLoader")
    static let models = URL.init(filePath: "/Volumes/Vol2/models")
    let model: ModelInfo
    let computeUnits: ComputeUnits
    private let maxSeed: UInt32

    private var downloadSubscriber: Cancellable?

    init(model: ModelInfo, computeUnits: ComputeUnits? = nil, maxSeed: UInt32 = UInt32.max) {
        self.model = model
        self.computeUnits = computeUnits ?? model.defaultComputeUnits
        self.maxSeed = maxSeed
        state = .undetermined
        setInitialState()
    }

    enum PipelinePreparationPhase {
        case undetermined
        case waitingToDownload
        case downloading(Double)
        case downloaded
        case uncompressing
        case readyOnDisk
        case loaded
        case failed(Error)
    }

    var state: PipelinePreparationPhase {
        didSet {
            statePublisher.value = state
        }
    }
    private(set) lazy var statePublisher: CurrentValueSubject<PipelinePreparationPhase, Never> = CurrentValueSubject(state)
    private(set) var downloader: Downloader? = nil

    func setInitialState() {
        if ready {
            state = .readyOnDisk
            return
        }
        if downloaded {
            state = .downloaded
            return
        }
        state = .waitingToDownload
    }
}

extension PipelineLoader {
    func cancel() { downloader?.cancel() }
}

extension PipelineLoader {
    var url: URL {
        return model.modelURL(for: variant)
    }

    var filename: String {
        return url.lastPathComponent
    }

    var downloadedURL: URL { PipelineLoader.models.appendingPathComponent(filename) }

    var uncompressURL: URL { PipelineLoader.models }

    var packagesFilename: String { (filename as NSString).deletingPathExtension }

    var compiledURL: URL { downloadedURL.deletingLastPathComponent().appendingPathComponent(packagesFilename)  }

    var downloaded: Bool {
        return FileManager.default.fileExists(atPath: downloadedURL.path)
    }

    var ready: Bool {
        return FileManager.default.fileExists(atPath: compiledURL.path)
    }

    var variant: AttentionVariant {
        switch computeUnits {
        case .cpuOnly           : return .original          // Not supported yet
        case .cpuAndGPU         : return .original
        case .cpuAndNeuralEngine: return model.supportsAttentionV2 ? .splitEinsumV2 : .splitEinsum
        case .all               : return model.isSD3 ? .original : .splitEinsum
        @unknown default:
            fatalError("Unknown MLComputeUnits")
        }
    }

    func prepare() async throws -> Pipeline {
        do {
            do {
                try FileManager.default.createDirectory(atPath: PipelineLoader.models.path, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating PipelineLoader.models path: \(error)")
            }

            try await download()
            try await unzip()
            let pipeline = try await load(url: compiledURL)
            return Pipeline(pipeline, maxSeed: maxSeed)
        } catch {
            state = .failed(error)
            logger.warning("Pipeline loading failed: \(error)")
            throw error
        }
    }

    @discardableResult
    func download() async throws -> URL {
        if ready || downloaded { return downloadedURL }

        let downloader = Downloader(from: url, to: downloadedURL)
        self.downloader = downloader
        downloadSubscriber = downloader.downloadState.sink { state in
            if case .downloading(let progress) = state {
                self.state = .downloading(progress)
            }
        }
        try downloader.waitUntilDone()
        logger.info("Download of \(self.url) complete")
        return downloadedURL
    }

    func unzip() async throws {
        guard downloaded else { return }
        state = .uncompressing
        do {
            try FileManager().unzipItem(at: downloadedURL, to: uncompressURL)
        } catch {
            logger.warning("Unzip of \(self.downloadedURL) failed")
            throw error
        }
        state = .readyOnDisk
        logger.info("Unzip of \(self.downloadedURL) complete")
    }

    func load(url: URL) async throws -> StableDiffusionPipelineProtocol {
        let beginDate = Date()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let pipeline: StableDiffusionPipelineProtocol
        if model.isXL {
            if #available(macOS 14.0, iOS 17.0, *) {
                pipeline = try StableDiffusionXLPipeline(resourcesAt: url,
                                                         configuration: configuration,
                                                         reduceMemory: model.reduceMemory)
            } else {
                logger.warning("Stable Diffusion XL requires macOS 14")
                throw "Stable Diffusion XL requires macOS 14"
            }

        } else if model.isSD3 {
            if #available(macOS 14.0, iOS 17.0, *) {
                pipeline = try StableDiffusion3Pipeline(resourcesAt: url,
                                                        configuration: configuration,
                                                        reduceMemory: model.reduceMemory)
            } else {
                logger.warning("Stable Diffusion 3 requires macOS 14")
                throw "Stable Diffusion 3 requires macOS 14"
            }
        } else {
            pipeline = try StableDiffusionPipeline(resourcesAt: url,
                                                       controlNet: [],
                                                       configuration: configuration,
                                                       disableSafety: false,
                                                       reduceMemory: model.reduceMemory)
        }
        try pipeline.loadResources()
        state = .loaded
        logger.info("Pipeline loaded in \(Date().timeIntervalSince(beginDate))")
        return pipeline
    }
}
