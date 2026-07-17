# Build context is the repo root (see docker-compose.yml) so the gateway image
# can bake in the Rust processor binary it spawns per video.
FROM rust:1.79-slim AS rust
WORKDIR /p
COPY processor-rust/Cargo.toml ./
COPY processor-rust/src ./src
RUN cargo build --release --bin vidgrep-processor

FROM golang:1.22 AS go
WORKDIR /g
COPY gateway-go/go.mod gateway-go/go.sum ./
RUN go mod download
COPY gateway-go/ ./
RUN CGO_ENABLED=0 go build -trimpath -o /vidgrep-gateway .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=rust /p/target/release/vidgrep-processor /usr/local/bin/vidgrep-processor
COPY --from=go /vidgrep-gateway /usr/local/bin/vidgrep-gateway
ENV VIDGREP_GATEWAY_ADDR=:8080 VIDGREP_PROCESSOR_BIN=vidgrep-processor
EXPOSE 8080
ENTRYPOINT ["vidgrep-gateway"]
