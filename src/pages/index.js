import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

export default function Home() {
  return (
    <Layout
      title="LeanPass"
      description="A lightweight, NumPy-only reverse-mode autodiff library for learning what's actually happening inside backprop."
    >
      <main>
        <div className="hero hero--leanpass">
          <div className="container">
            <h1>LeanPass</h1>
            <p className="hero__subtitle">
              A tiny, NumPy-only reverse-mode autodiff library.
              <br />
              Not competing with PyTorch on features, competing on{' '}
              <em>being lightweight and readable!</em>.
            </p>
            <CodeBlock language="bash">pip install leanpass</CodeBlock>

            <div style={{ marginTop: '3rem', textAlign: 'center' }}>
              <img
                src="/img/computation_graph.png"
                alt="Computation graph showing forward and backward pass"
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: '8px',
                  border: '1px solid var(--ifm-color-emphasis-200)',
                }}
              />
              <p
                style={{
                  fontSize: '0.85rem',
                  color: 'var(--ifm-color-emphasis-600)',
                  marginTop: '0.5rem',
                }}
              >
                A small computation graph: forward pass, backward pass.
              </p>
            </div>

            <div
              style={{
                maxWidth: '650px',
                margin: '3rem auto 0',
                textAlign: 'left',
                lineHeight: 1.75,
              }}
            >
              <p>
                LeanPass exists because most autodiff libraries bury the mechanics
                under C++ kernels and JIT compilers. If you're trying to understand
                <em>how</em> gradients flow through a neural network not just
                <em>that</em> they flow, you need source code that fits in your head.
              </p>
              <p>
                This is that library. ~730 lines of pure NumPy. Every
                <code>backward()</code> call traces through the graph step by step.
                No magic, no abstraction layers, just vector calculus you can read
                line by line.
              </p>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
