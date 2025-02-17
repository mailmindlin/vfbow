use std::{fs::File, io::Read};

use vfbow::{Vocabulary, VocabularyReadOptions};
use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("read_buffer", |b| {
        let buffer = {
            let mut file = File::open("./data/orb_mur.fbow").unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();
            buffer
        };

        b.iter(|| {
            let mut read: &[u8] = &buffer;
            Vocabulary::read_from(&mut read, Default::default()).unwrap()
        });
    });

    c.bench_function("read_file", |b| {
        let options = VocabularyReadOptions::all(vfbow::ParseValidationMode::Ignore);
        b.iter_batched(
            || File::open("./data/orb_mur.fbow").unwrap(),
            |file| {
                Vocabulary::read_from(file, options).unwrap()
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
